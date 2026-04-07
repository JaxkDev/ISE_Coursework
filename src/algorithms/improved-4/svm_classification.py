# Source improved-2/svm_classification
#
# Modified to use pre-trained word embeddings (Word2Vec) instead of TF-IDF features.

ID = 4

SRC_DIR = "./src/algorithms/improved-"+str(ID)+"/"
TMP_DIR = SRC_DIR + "tmp/"
RESULTS_DIR = "./results/improved-"+str(ID)+"/"
DATASET_DIR = "./dataset/"

PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe', 'all']
REPEAT_TIMES = [10, 20, 50]

########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import time
import re

# For word embeddings
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler

# glove-wiki-gigaword-300 (smaller, GloVe) or word2vec-google-news-300 (larger, Word2Vec)
word2vec_vectors = api.load('word2vec-google-news-300') #https://huggingface.co/fse/word2vec-google-news-300/

def words_to_embedding(text):
    words = text.split()
    word_vectors = []
    
    for word in words:
        try:
            word_vectors.append(word2vec_vectors[word])
        except KeyError:
            continue
    
    # If no known words, return a zero vector of dimension 300
    if not word_vectors:
        return np.zeros(word2vec_vectors.vector_size)  # 300
    else:
        return np.mean(word_vectors, axis=0)

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.svm import LinearSVC

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

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

########## 3. Download & read data ##########
import os

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

for project in PROJECTS:
    pd_all = pd.DataFrame()

    if project == 'all':
        # If 'all', read and concatenate all project files
        for proj in ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']:
            path = f'{proj}.csv'
            pd_temp = pd.read_csv("./dataset/"+path)
            pd_all = pd.concat([pd_all, pd_temp], ignore_index=True)
            pd_all = pd_all.sample(frac=1, random_state=51003)  # Shuffle
    else:
        path = f'{project}.csv'
        pd_all = pd.read_csv("./dataset/"+path)
        pd_all = pd_all.sample(frac=1, random_state=51003)  # Shuffle

    # Merge Title and Body into a single column; if Body is NaN, use Title only
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    pd_tplusb.to_csv(TMP_DIR + project + '_Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

    ########## 4. Configure parameters & Start training ##########

    # ========== Key Configurations ==========

    # 1) Data file to read
    datafile = TMP_DIR + project + '_Title+Body.csv'

    # 3) Output CSV file name
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    out_csv_name = f'{RESULTS_DIR}{project}_Word2Vec_SVM.csv'

    # ========== Read and clean data ==========
    data = pd.read_csv(datafile).fillna('')
    text_col = 'text'

    # Keep a copy for referencing original data if needed
    original_data = data.copy()

    # Text cleaning
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)

    # ========== Hyperparameter grid ==========
    # We use logspace for var_smoothing: [1e-12, 1e-11, ..., 1]
    params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization
    }


    for REPEAT in REPEAT_TIMES:
        print(f"\n--- [Improved-{ID}] Running Linear SVM + Word2Vec for project: {project} with {REPEAT} repeats ---")
        # Lists to store metrics across repeated runs
        accuracies  = []
        precisions  = []
        recalls     = []
        f1_scores   = []
        auc_values  = []
        mcc_values  = []
        cm_sum = np.array([[0, 0], [0, 0]])  # Initialize confusion matrix sum
        train_times = []
        pred_times  = []
        for repeated_time in range(REPEAT):
            # --- 4.1 Split into train/test ---
            indices = np.arange(data.shape[0])
            train_index, test_index = train_test_split(
                indices, test_size=0.3, random_state=repeated_time, stratify=data['sentiment'] #class balance stratification
            )

            train_text = data[text_col].iloc[train_index]
            test_text = data[text_col].iloc[test_index]

            y_train = data['sentiment'].iloc[train_index]
            y_test  = data['sentiment'].iloc[test_index]

            # --- 4.2 TF-IDF vectorization ---
            X_train = np.array([
                words_to_embedding(text) 
                for text in train_text
            ])

            X_test = np.array([
                words_to_embedding(text) 
                for text in test_text
            ])

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
                
            # --- 4.3 Linear SVC model & GridSearch ---
            clf = LinearSVC(max_iter=20000, class_weight='balanced', random_state=51003)  # Increase max_iter to avoid warnings
            grid = GridSearchCV(
                clf,
                params,
                n_jobs=-1,         # Use all available cores
                cv=5,              # 5-fold CV (can be changed)
                scoring='f1_macro' # Using f1_macro as the metric for selection
            )
            grid.fit(X_train, y_train)

            # Train the best model (for timings, we could just use best estimator straight away but to be consistent for timing recordings, we retrain it here)
            TRAIN_TIME = time.perf_counter_ns()
            best_clf = LinearSVC(**grid.best_params_, max_iter=20000, class_weight='balanced', random_state=51003)
            best_clf.fit(X_train, y_train)

            TRAIN_TIME = time.perf_counter_ns() - TRAIN_TIME
            train_times.append(TRAIN_TIME)

            # --- 4.4 Make predictions & evaluate ---
            PRED_TIME = time.perf_counter_ns()

            y_pred = best_clf.predict(X_test)

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
            y_score = best_clf.decision_function(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc_val = auc(fpr, tpr)
            auc_values.append(auc_val)

            # MCC
            mcc = matthews_corrcoef(y_test, y_pred)
            mcc_values.append(mcc)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred) # [[tn, fp], [fn, tp]]
            cm_sum += cm

        # --- 4.5 Aggregate results ---
        final_accuracy  = np.mean(accuracies)
        final_precision = np.mean(precisions)
        final_recall    = np.mean(recalls)
        final_f1        = np.mean(f1_scores)
        final_auc       = np.mean(auc_values)
        final_mcc       = np.mean(mcc_values)
        final_cm        = cm_sum / REPEAT  # Average confusion matrix
        final_train_time = np.mean(train_times, dtype=np.int64)
        final_pred_time  = np.mean(pred_times, dtype=np.int64)

        #print("=== Naive Bayes + TF-IDF Results ===")
        print(f"Number of repeats:       {REPEAT}")
        print(f"Average Training Time:   {final_train_time}")
        print(f"Average Prediction Time: {final_pred_time}")
        print(f"Average Accuracy:        {final_accuracy:.4f}")
        print(f"Average Precision:       {final_precision:.4f}")
        print(f"Average Recall:          {final_recall:.4f}")
        print(f"Average F1 score:        {final_f1:.4f}")
        print(f"Average AUC:             {final_auc:.4f}")
        print(f"Average MCC:             {final_mcc:.4f}")
        print(f"Average Confusion Matrix:\n{final_cm}")


        # Save final results to CSV (append mode)
        try:
            # Attempt to check if the file already has a header
            existing_data = pd.read_csv(out_csv_name, nrows=1)
            header_needed = False
        except:
            header_needed = True

        df_log = pd.DataFrame(
            {
                'repeated_times': [REPEAT],
                'Training_Time': [final_train_time],
                'Prediction_Time': [final_pred_time],
                'Accuracy': [final_accuracy],
                'Precision': [final_precision],
                'Recall': [final_recall],
                'F1': [final_f1],
                'AUC': [final_auc],
                'MCC': [final_mcc],
                'CM': [final_cm.tolist()],  # Store confusion matrix as a list [[tn, fp], [fn, tp]]
                'CV_list(AUC)': [str(auc_values)]
            }
        )

        df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

    print(f"\nResults have been saved to: {out_csv_name}")

