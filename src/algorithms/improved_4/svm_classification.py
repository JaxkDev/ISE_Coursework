# Source improved-2/svm_classification
#
# Modified to use pre-trained word embeddings (Word2Vec) instead of TF-IDF features.



import numpy as np
import time
import re

from tqdm import tqdm

# Text and feature engineering
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.svm import LinearSVC

# Base class
from src.algorithms.algorithms import BaseAlgorithm

from typing import Optional, Any


class SVMAlgorithm(BaseAlgorithm):

    def __init__(self) -> None:
        """Initialise the model with no data or internal state."""
        super().__init__()
        self.vectors = None
        self.vectors_model = 'word2vec-google-news-300'  #https://huggingface.co/fse/word2vec-google-news-300/
        # glove-wiki-gigaword-300 (smaller, GloVe) or word2vec-google-news-300 (larger, Word2Vec)

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

        
        self.vectors = api.load(self.vectors_model)

    def words_to_embedding(self, text):
        words = text.split()
        word_vectors = []
        
        for word in words:
            try:
                word_vectors.append(self.vectors[word])
            except KeyError:
                continue
        
        # If no known words, return a zero vector of dimension 300
        if not word_vectors:
            return np.zeros(self.vectors.vector_size)  # 300
        else:
            return np.mean(word_vectors, axis=0)

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
            # Train-test split (70/30 split, stratified by class)
            indices = np.arange(self.data.shape[0])
            train_index, test_index = train_test_split(
                indices, test_size=0.3, random_state=repeated_time, stratify=self.data['sentiment'] #class balance stratification
            )

            train_text = self.data['text'].iloc[train_index]
            test_text = self.data['text'].iloc[test_index]

            y_train = self.data['sentiment'].iloc[train_index]
            y_test  = self.data['sentiment'].iloc[test_index]

            # TF-IDF vectorization
            X_train = np.array([
                self.words_to_embedding(text) 
                for text in train_text
            ])

            X_test = np.array([
                self.words_to_embedding(text) 
                for text in test_text
            ])

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            params = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization
            }
        
            clf = LinearSVC(max_iter=20000, class_weight='balanced', random_state=seed)
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

            self.model = LinearSVC(**grid.best_params_, max_iter=20000, class_weight='balanced', random_state=seed)
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
