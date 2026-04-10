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



import numpy as np
import torch
import time
import re
import os

from tqdm import tqdm

# Evaluation and tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score,
                             f1_score, roc_auc_score)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

# Base class
from src.algorithms.algorithms import BaseAlgorithm

from typing import Optional, Any


MODELS_DIR = os.path.dirname(os.path.abspath(__file__))+"/models/"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


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


class BERTAlgorithm(BaseAlgorithm):

    def __init__(self) -> None:
        super().__init__()
        self.trained = False
        self.trainer = None
        self.tokenizer = None
        self.data_collator = None
        self.model_name = 'bert-base-uncased'  # or 'microsoft/codebert-base' (Roberta-based, trained on code)

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
        self.data['text'] = self.data['text'].apply(clean_str)


    def load_model(self, fresh: bool = False) -> None:
        self.fresh = fresh
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if not fresh:
            model_path = f'{MODELS_DIR}{self.project}/'
            if os.path.exists(model_path):
                print(f"Loading pre-trained model for {self.project} from {model_path}")
                self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
                self.trained = True
            else:
                print(f"No pre-trained model found for {self.project} at {model_path}. Training a new model.")
                self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        else:
            print(f"Fresh load requested. Initialising new model for {self.project} without loading pre-trained weights.")
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

    def train(self, repetitions: int = 1, seed: Optional[int] = 51003) -> dict[str, Any]:
        """
        Train the model using the loaded data.

        Parameters
        ----------
        repetitions : int, default=1
            The number of times to repeat the training process.
        seed : int or None, default=51003
            Optional random seed for reproducibility.

        Returns
        -------
        metrics : dict
            A dictionary containing evaluation metrics (e.g., accuracy, precision) for the trained model on the test set over all repetitions.
        """
        if self.data is None or self.data_collator is None or self.tokenizer is None:
            raise ValueError("No data loaded. Please load a dataset before training.")
        
        if repetitions != 1:
            print("======= Warning: Repeating training for BERT is very time-consuming. Consider setting repetitions=1 for faster results =======")
            print("To akcowledge this warning and proceed with the specified repetitions, press enter.\n")
            input()
        
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

            # Prepare data
            train_texts = train_text.tolist()
            test_texts = test_text.tolist()
            train_labels = y_train.tolist()
            test_labels = y_test.tolist()

            train_dataset = BugReportDataset(train_texts, train_labels, self.tokenizer)
            test_dataset = BugReportDataset(test_texts, test_labels, self.tokenizer)

            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )

            class_weights = torch.tensor(class_weights, dtype=torch.float)

            class WeightedTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.get("logits")

                    loss_fct = CrossEntropyLoss(weight=class_weights.to(model.device))
                    loss = loss_fct(logits, labels)

                    return (loss, outputs) if return_outputs else loss

            # Training arguments
            training_args = TrainingArguments(
                output_dir='tmp',  # output directory
                num_train_epochs=3,
                eval_strategy="epoch",
                save_strategy="no", # effectively disable intermediate saving, we will save manually after training
                report_to="none",
                learning_rate=2e-5,
                do_eval=True,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                seed=seed,
                data_seed=seed,
            )

            # Trainer
            self.trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=self.data_collator,
            )

            # Train the best model (for timings, we could just use best estimator straight away but to be consistent for timing recordings, we retrain it here)
            TRAIN_TIME = time.perf_counter_ns()

            # Train
            if not self.trained:
                self.trainer.train()
            else:
                print(f"Skipping training for {self.project} as pre-trained model is loaded.")

            TRAIN_TIME = time.perf_counter_ns() - TRAIN_TIME if not self.trained else 0  # If model was pre-trained and loaded, set training time to 0
            train_times.append(TRAIN_TIME)

            if self.fresh or not self.trained:
                # Save the trained model for this project
                save_path = f'{MODELS_DIR}{self.project}/'
                self.trainer.save_model(save_path)
                print(f"Model for {self.project} saved to {save_path}")

            PRED_TIME = time.perf_counter_ns()

            # Evaluate on the test set
            predictions = self.trainer.predict(test_dataset)

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
            probs = softmax(logits, axis=1)   # convert to probabilities
            auc_val = roc_auc_score(y_test, probs[:, 1])  # AUC for positive class
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
        raise NotImplementedError("Using BERTAlgorithm in CLI is out of scope for this project.")
