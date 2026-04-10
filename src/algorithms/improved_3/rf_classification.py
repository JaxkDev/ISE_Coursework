# Source improved-2/svm_classification
#
# Modified to Random Forest Classifier instead of Naive Bayes.


import numpy as np
import time
import re

from tqdm import tqdm

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.ensemble import RandomForestClassifier

# Base class
from src.algorithms.algorithms import BaseAlgorithm

from typing import Optional, Any


class RFAlgorithm(BaseAlgorithm):

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

        # Stopwords
        import nltk
        nltk.download('stopwords', quiet=True) # here because of UI log being annoying with the download message every time.
        from nltk.corpus import stopwords

        NLTK_stop_words_list = stopwords.words('english')
        custom_stop_words_list = ['...']  # You can customize this list as needed
        final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

        def remove_stopwords(text):
            """Remove stopwords from the text."""
            return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

        def clean_str(string):
            """
            Clean text by removing non-alphanumeric characters,
            and convert it to lowercase.
            """
            string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"\)", " ) ", string)
            string = re.sub(r"\?", " ? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            string = re.sub(r"\\", "", string)
            string = re.sub(r"\'", "", string)
            string = re.sub(r"\"", "", string)
            return string.strip().lower()
    
        self.data['text'] = self.data['text'].apply(remove_html)
        self.data['text'] = self.data['text'].apply(remove_emoji)
        self.data['text'] = self.data['text'].apply(remove_stopwords)
        self.data['text'] = self.data['text'].apply(clean_str)


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
            # Train/test split (70/30 split, stratified by class)
            indices = np.arange(self.data.shape[0])
            train_index, test_index = train_test_split(
                indices, test_size=0.3, random_state=repeated_time, stratify=self.data['sentiment'] #class balance stratification
            )

            train_text = self.data['text'].iloc[train_index]
            test_text = self.data['text'].iloc[test_index]

            y_train = self.data['sentiment'].iloc[train_index]
            y_test  = self.data['sentiment'].iloc[test_index]

            # TF-IDF vectorization
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000
            )
            X_train = tfidf.fit_transform(train_text)
            X_test = tfidf.transform(test_text)

            params = {
                'n_estimators': [100, 200],
                'max_features': ['sqrt', 'log2'],
                #'min_samples_leaf': [1, 2, 5], Removed to reduce tuning time; can be added back if time allows. (does not improve- dont add backl)
                #'min_samples_split': [2, 5]
            }
        
            clf = RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=1) 
            grid = GridSearchCV(
                clf,
                params,
                n_jobs=1,          # Must be 1 for reproducibility in timing; set to -1 for faster tuning if reproducibility is not a concern
                cv=5,              # 5-fold CV (can be changed)
                scoring='f1_macro' # Using f1_macro as the metric for selection
            )
            grid.fit(X_train, y_train)

            # Train the best model (for timings, we could just use best estimator straight away but to be consistent for timing recordings, we retrain it here)
            TRAIN_TIME = time.perf_counter_ns()

            self.model = RandomForestClassifier(**grid.best_params_, class_weight='balanced', random_state=seed, n_jobs=1)
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
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            precisions.append(prec)

            # Recall (macro)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            recalls.append(rec)

            # F1 Score (macro)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
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
        
        return self.model.predict([X])[0]
