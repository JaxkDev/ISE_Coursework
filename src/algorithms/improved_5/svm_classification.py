# Source improved-2/svm_classification
#
# Modified to NOT remove stopwords AND lemmatize text using WordNetLemmatizer.
# And use sublinear_tf=True in TfidfVectorizer to improve performance and handle term frequency better.



import numpy as np
import time
import re

from tqdm import tqdm

# Text and feature engineering
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_curve, auc)

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

# Classifier
from sklearn.svm import LinearSVC

# Base class
from src.algorithms.algorithms import BaseAlgorithm

from typing import Optional, Any


class SVMAlgorithm(BaseAlgorithm):

    def preprocess_data(self) -> None:
        if self.data is None:
            raise ValueError("No data loaded. Please load a dataset before preprocessing.")
        
        def remove_html(text):
            """Remove HTML tags using a regex."""
            html = re.compile(r'<.*?>')
            return html.sub(r'', text)

        def remove_emoji(text):
            """Remove emojis using a regex pattern."""
            emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"  # enclosed characters
                                    "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', text)

        def lowercase_text(text):
            """Convert text to lowercase."""
            return text.lower()

        nltk.download('wordnet', quiet=True) # here because of UI log being annoying with the download message every time.
        lemmatizer = WordNetLemmatizer()

        def lemmatize_text(text):
            """Lemmatize text using WordNetLemmatizer."""
            return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        
        self.data['text'] = self.data['text'].apply(remove_html)
        self.data['text'] = self.data['text'].apply(remove_emoji)
        self.data['text'] = self.data['text'].apply(lowercase_text)
        self.data['text'] = self.data['text'].apply(lemmatize_text)


    def load_model(self, fresh: bool = True) -> None:
        pass  # No model to load for SVM.


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
            # Train/test split (70/30) with stratification for class balance
            indices = np.arange(self.data.shape[0])
            train_index, test_index = train_test_split(
                indices, test_size=0.3, random_state=repeated_time, stratify=self.data['sentiment'] #class balance stratification
            )

            train_text = self.data['text'].iloc[train_index]
            test_text = self.data['text'].iloc[test_index]

            y_train = self.data['sentiment'].iloc[train_index]
            y_test  = self.data['sentiment'].iloc[test_index]
        
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )
            X_train = tfidf.fit_transform(train_text)
            X_test = tfidf.transform(test_text)

            pipeline = Pipeline([
                ('chi2', SelectKBest(chi2)),
                ('svc', LinearSVC(
                    max_iter=5000, 
                    class_weight='balanced', 
                    random_state=seed
                ))
            ])

            # GridSearch now tunes chi2 k AND SVM C together:
            params = {
                'chi2__k': [100, 250, 500, 750, 1000],
                'svc__C': [0.001, 0.01, 0.1, 1, 10, 100],
            }
        
            # Linear SVC model & GridSearch
            grid = GridSearchCV(pipeline, params, cv=5, scoring='f1_macro', n_jobs=1) # Must be 1 for reproducibility in timing; set to -1 for faster tuning if reproducibility is not a concern
            grid.fit(X_train, y_train)

            # Train the best model (for timings, we could just use best estimator straight away but to be consistent for timing recordings, we retrain it here)
            TRAIN_TIME = time.perf_counter_ns()

            self.model = pipeline.set_params(**grid.best_params_)
            self.model.fit(X_train, y_train)

            TRAIN_TIME = time.perf_counter_ns() - TRAIN_TIME
            train_times.append(TRAIN_TIME)

            PRED_TIME = time.perf_counter_ns()

            # Evaluate
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
            y_score = self.model.decision_function(X_test)

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
        
        return self.model.predict([X])[0]
