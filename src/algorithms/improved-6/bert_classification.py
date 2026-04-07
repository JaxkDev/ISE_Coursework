# Source improved-5/svm_classification
#
# Modified to use BERT for classification instead of SVM.
#
# =========================================================================================================
#
# IMPORTANT MARKER NOTE:
#
# THIS TAKES A VERY VERY LONG TIME TO RUN (DUE TO ML TRAINING ON CPU)
# On my 16-Core 5GHz CPU, it takes around 25+min for ALL project data with NO repeats (set to 1 = no repeats).
#
# It should be usable with GPU but due to unknown system compatibility with whomever marks this project,
# I have opted to strictly use CPU only.
#
# =========================================================================================================

ID = 6

LOAD_MODEL = False  # Set to True if you want to load a pre-trained model (if available) instead of training from scratch.
# This will use the model already trained on each project and saved in the "models" folder. If False, it will train a new model for each project and overwrite the existing one.
# This is enabled by default to allow marker to quickly see results, but can be set to False to verify training if you so wish.

SRC_DIR = "./src/algorithms/improved-"+str(ID)+"/"
TMP_DIR = SRC_DIR + "tmp/"
RESULTS_DIR = "./results/improved-"+str(ID)+"/"
DATASET_DIR = "./dataset/"

PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe', 'all']
REPEAT_TIMES = [1]

########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import time
import re

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import torch
from scipy.special import softmax
from torch.utils.data import Dataset

# Use pre-trained BERT
model_name = 'bert-base-uncased'  # or 'microsoft/codebert-base' (Roberta-based, trained on code)
tokenizer = BertTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

class BugReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=False, 
                                   max_length=max_length)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


from sklearn.utils.class_weight import compute_class_weight

# Evaluation and tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_auc_score)

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

if not os.path.exists(SRC_DIR + "models"):
    os.makedirs(SRC_DIR + "models")

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
    out_csv_name = f'{RESULTS_DIR}{project}_BERT.csv'

    # ========== Read and clean data ==========
    data = pd.read_csv(datafile).fillna('')
    text_col = 'text'

    # Keep a copy for referencing original data if needed
    original_data = data.copy()

    # Text cleaning
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    #data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)


    for REPEAT in REPEAT_TIMES:
        print(f"\n--- [Improved-{ID}] Running BERT for project: {project} with {REPEAT} repeats ---")
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

            # Prepare data
            train_texts = train_text.tolist()
            test_texts = test_text.tolist()
            train_labels = y_train.tolist()
            test_labels = y_test.tolist()

            train_dataset = BugReportDataset(train_texts, train_labels, tokenizer)
            test_dataset = BugReportDataset(test_texts, test_labels, tokenizer)


            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )

            class_weights = torch.tensor(class_weights, dtype=torch.float)

            from torch.nn import CrossEntropyLoss
            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.get("logits")

                    loss_fct = CrossEntropyLoss(weight=class_weights.to(model.device))
                    loss = loss_fct(logits, labels)

                    return (loss, outputs) if return_outputs else loss

            
            # Load model
            TRAINED_ALREADY = False
            if LOAD_MODEL:
                model_path = f'{SRC_DIR}models/{project}/'
                if os.path.exists(model_path):
                    print(f"Loading pre-trained model for {project} from {model_path}")
                    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
                    TRAINED_ALREADY = True
                else:
                    print(f"No pre-trained model found for {project} at {model_path}. Training a new model.")
                    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            else:
                model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=TMP_DIR + 'model',  # output directory
                num_train_epochs=3,
                eval_strategy="epoch",
                save_strategy="no", # effectively disable intermediate saving, we will save manually after training
                report_to="none",
                learning_rate=2e-5,
                do_eval=True,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
            )

            # Trainer
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=data_collator,
            )

            # --- 4.3 Linear SVC model & GridSearch ---
            TRAIN_TIME = time.perf_counter_ns()

            # Train
            if not TRAINED_ALREADY:
                trainer.train()
            else:
                print(f"Skipping training for {project} as pre-trained model is loaded.")

            TRAIN_TIME = time.perf_counter_ns() - TRAIN_TIME
            train_times.append(TRAIN_TIME)

            if not LOAD_MODEL or not TRAINED_ALREADY:
                # Save the trained model for this project
                save_path = f'{SRC_DIR}models/{project}/'
                trainer.save_model(save_path)
                print(f"Model for {project} saved to {save_path}")

            # --- 4.4 Make predictions & evaluate ---
            PRED_TIME = time.perf_counter_ns()

            predictions = trainer.predict(test_dataset)

            PRED_TIME = time.perf_counter_ns() - PRED_TIME
            pred_times.append(PRED_TIME)

            logits = predictions.predictions
            y_pred = np.argmax(logits, axis=1)

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
            logits = predictions.predictions
            probs = softmax(logits, axis=1)   # convert to probabilities
            auc_val = roc_auc_score(y_test, probs[:, 1])  # AUC for positive class
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

