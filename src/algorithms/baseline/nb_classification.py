# Source https://github.com/JaxkDev/ISE_Lab1/blob/master/main.py
#
# Modifications:
# - Timer used (performance counter, nanoseconds) to measure training and prediction time more accurately.
# - Grid search now uses all available CPU cores (n_jobs=-1) to speed up hyperparameter tuning.
# - Fixed AUC calculation to use predicted probabilities instead of class labels.
#
# Modified to coursework folder structure:
# ./baseline/br_classification.py
# ./dataset/pytorch.csv, etc.
# ./results/baseline/pytorch_NB.csv, etc.



import numpy as np
import time

from tqdm import tqdm

# Text and feature engineering
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.naive_bayes import GaussianNB

# Base class
from src.algorithms.algorithms import BaseAlgorithm

from typing import Optional, Any


class NBAlgorithm(BaseAlgorithm):

    def __init__(self):
        super().__init__()
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        self.stop_words = stopwords.words('english') + ['...']  # You can customize this list as needed

    def remove_stopwords(self, text):
        """Remove stopwords from the text."""
        return " ".join([word for word in str(text).split() if word not in self.stop_words])

    def preprocess_data(self) -> None:
        if self.data is None:
            raise ValueError("No data loaded. Please load a dataset before preprocessing.")
    
        self.data['text'] = self.data['text'].apply(self.remove_html)
        self.data['text'] = self.data['text'].apply(self.remove_emoji)
        self.data['text'] = self.data['text'].apply(self.remove_stopwords)
        self.data['text'] = self.data['text'].apply(self.clean_str)


    def load_model(self, fresh: bool = True) -> None:
        pass  # No model to load


    def train(self, repetitions: int = 10, seed: Optional[int] = 51003) -> dict[str, Any]:
        if self.data is None:
            raise ValueError("No data loaded. Please load a dataset before training.")
        
        # Lists to store metrics across repeated runs
        accuracies  = []
        precisions  = []
        recalls     = []
        f1_scores   = []
        auc_values  = []
        mcc_values  = []
        cm_values   = []
        train_times = []
        pred_times  = []
        for repeated_time in tqdm(range(repetitions), desc=f"Training {repetitions} repetitions"):
    
            # Split into train/test (70/30 split, stratified by class)
            indices = np.arange(self.data.shape[0])
            train_index, test_index = train_test_split(
                indices,
                test_size=0.3,
                random_state=repeated_time,
                stratify=self.data['sentiment'] #class balance stratification
            )

            train_text = self.data['text'].iloc[train_index]
            test_text = self.data['text'].iloc[test_index]

            y_train = self.data['sentiment'].iloc[train_index]
            y_test  = self.data['sentiment'].iloc[test_index]

            # TF-IDF vectorization
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000
            )
            X_train = self.tfidf.fit_transform(train_text).toarray()
            X_test = self.tfidf.transform(test_text).toarray()

            params = {
                'var_smoothing': np.logspace(-12, 0, 13)
            }
        
            clf = GaussianNB()
            grid = GridSearchCV(
                clf,
                params,
                n_jobs=-1,         # Use all available CPU cores for parallel processing
                cv=5,              # 5-fold CV (can be changed)
                scoring='f1_macro' # Using f1_macro as the metric for selection
            )
            grid.fit(X_train, y_train)

            # Train the best model (for timings, we could just use best estimator straight away but to be consistent for timing recordings, we retrain it here)
            TRAIN_TIME = time.perf_counter_ns()

            self.model = GaussianNB(**grid.best_params_)
            self.model.fit(X_train, y_train)

            TRAIN_TIME = time.perf_counter_ns() - TRAIN_TIME
            train_times.append(TRAIN_TIME)

            PRED_TIME = time.perf_counter_ns()

            # Evaluate on test set
            y_pred = self.model.predict(X_test)

            PRED_TIME = time.perf_counter_ns() - PRED_TIME
            pred_times.append(PRED_TIME)

            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

            # Precision (macro)
            prec = precision_score(y_test, y_pred, average='macro')
            precisions.append(prec)

            # Recall (macro)
            rec = recall_score(y_test, y_pred, average='macro')
            recalls.append(rec)

            # F1 Score (macro)
            f1 = f1_score(y_test, y_pred, average='macro')
            f1_scores.append(f1)

            # AUC
            y_score = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_val = auc(fpr, tpr)
            auc_values.append(auc_val)

            # MCC
            mcc = matthews_corrcoef(y_test, y_pred)
            mcc_values.append(mcc)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred) # [[tn, fp], [fn, tp]]
            cm_values.append(cm)

        return {
            'training_time': train_times,
            'prediction_time': pred_times,
            'accuracy': accuracies,
            'precision': precisions,
            'recall': recalls,
            'f1': f1_scores,
            'auc': auc_values,
            'mcc': mcc_values,
            'cm': cm_values
        }


    def predict(self, X: str) -> int:
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model before making predictions.")

        # clean
        X = self.remove_html(X)
        X = self.remove_emoji(X)
        X = self.remove_stopwords(X)
        X = self.clean_str(X)

        X_vec = self.tfidf.transform([X]).toarray()

        return self.model.predict(X_vec)[0]
