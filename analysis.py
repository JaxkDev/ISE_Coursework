######################
##   ANALYSIS CLI   ##
######################

# This file contains the code for reproducing the experiments/results in the coursework report.

import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from src.algorithms import get_algorithm

# get CLI arguments for which algorithm and project to run, and how many repetitions to do.
get_parser = ArgumentParser(description="Analyze experiment results for a given algorithm and project.")
get_parser.add_argument("--algorithm", type=str, default='*', help="The algorithm to analyse (e.g., 'baseline', 'improved_1', etc. * to run all)", choices=['baseline', 'improved_1', 'improved_2', 'improved_3', 'improved_4', 'improved_5', 'improved_6', 'final', '*'])
get_parser.add_argument("--project", type=str, default='*', help="The project/dataset to analyse (e.g., 'pytorch', 'tensorflow', etc. * to run all)", choices=['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe', 'all', '*'])
get_parser.add_argument("--repetitions", type=int, default=-1, help="The number of repetitions to perform for the training process. -1 to run (10, 20, 50) for each algorithm and project.")
get_parser.add_argument("--cached_models", type=str, default='True', help="Whether to load cached models from disk if available (True/False). If True, will attempt to load previously saved models. If False, will always train new models from scratch.", choices=['True', 'False', 'true', 'false'])
args = get_parser.parse_args()

algorithms_to_run = [args.algorithm] if args.algorithm != '*' else ['baseline', 'improved_1', 'improved_2', 'improved_3', 'improved_4', 'improved_5', 'improved_6', 'final']
projects_to_run = [args.project] if args.project != '*' else ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe', 'all']
repetitions_to_run = [args.repetitions] if args.repetitions != -1 else [10, 20, 50]
cached_models_to_use = args.cached_models.lower() == 'true'

print(f"\nRunning analysis for:\nAlgorithms: {', '.join(algorithms_to_run)},\nProjects: {', '.join(projects_to_run)},\nRepetitions: {', '.join(str(r) for r in repetitions_to_run)}\nUsing Cached Models: {'Yes' if cached_models_to_use else 'No'}\n\n")

for algorithm in algorithms_to_run:
    print(f"Analyzing algorithm: {algorithm}")
    for project in projects_to_run:
        print(f"Analyzing project: {algorithm}.{project}")

        algo = get_algorithm(algorithm)
        algo.load_dataset(project)
        algo.preprocess_data()

        # Summary stats for csv file
        rows = {
            'Repeated_Times': [],
            'Training_Time_Mean': [],
            'Prediction_Time_Mean': [],
            'Accuracy_Mean': [],
            'Accuracy_Median': [],
            'Accuracy_Q1': [],
            'Accuracy_Q3': [],
            'Precision_Mean': [],
            'Precision_Median': [],
            'Precision_Q1': [],
            'Precision_Q3': [],
            'Recall_Mean': [],
            'Recall_Median': [],
            'Recall_Q1': [],
            'Recall_Q3': [],
            'F1_Mean': [],
            'F1_Median': [],
            'F1_Q1': [],
            'F1_Q3': [],
            'AUC_Mean': [],
            'AUC_Median': [],
            'AUC_Q1': [],
            'AUC_Q3': [],
            'MCC_Mean': [],
            'MCC_Median': [],
            'MCC_Q1': [],
            'MCC_Q3': [],
            'CM_Mean': []
        }

        temp_repetitions = repetitions_to_run

        if algorithm == 'improved_6' and len(repetitions_to_run) > 1:  # Due to the time-consuming nature of training BERT models, we will only run 1 repetition for improved_6 in the all-projects analysis. For individual project analysis, we can run multiple repetitions.
            temp_repetitions = [1]
            print("Only doing one repetition for improved_6 due to time constraints of training BERT models for each project.")

        for repeat in temp_repetitions:
            algo.load_model(fresh=not cached_models_to_use)
            metrics = algo.train(repetitions=repeat, seed=51003)

            print(f"Metrics for {algorithm} on {project} (repeated {repeat} times):")
            print(f"Average Training Time:   {np.mean(metrics['training_time']) / 1e9:.8f} seconds")
            print(f"Average Prediction Time: {np.mean(metrics['prediction_time']) / 1e9:.8f} seconds")
            print(f"Average Accuracy:        {np.mean(metrics['accuracy']):.4f} (median: {np.median(metrics['accuracy']):.4f}, Q1: {np.percentile(metrics['accuracy'], 25):.4f}, Q3: {np.percentile(metrics['accuracy'], 75):.4f})")
            print(f"Average Precision:       {np.mean(metrics['precision']):.4f} (median: {np.median(metrics['precision']):.4f}, Q1: {np.percentile(metrics['precision'], 25):.4f}, Q3: {np.percentile(metrics['precision'], 75):.4f})")
            print(f"Average Recall:          {np.mean(metrics['recall']):.4f} (median: {np.median(metrics['recall']):.4f}, Q1: {np.percentile(metrics['recall'], 25):.4f}, Q3: {np.percentile(metrics['recall'], 75):.4f})")
            print(f"Average F1 Score:        {np.mean(metrics['f1']):.4f} (median: {np.median(metrics['f1']):.4f}, Q1: {np.percentile(metrics['f1'], 25):.4f}, Q3: {np.percentile(metrics['f1'], 75):.4f})")
            print(f"Average AUC:             {np.mean(metrics['auc']):.4f} (median: {np.median(metrics['auc']):.4f}, Q1: {np.percentile(metrics['auc'], 25):.4f}, Q3: {np.percentile(metrics['auc'], 75):.4f})")
            print(f"Average MCC:             {np.mean(metrics['mcc']):.4f} (median: {np.median(metrics['mcc']):.4f}, Q1: {np.percentile(metrics['mcc'], 25):.4f}, Q3: {np.percentile(metrics['mcc'], 75):.4f})")
            print(f"Average Confusion Matrix:\n{np.mean([np.array(cm) for cm in metrics['cm']], axis=0)}\n")


            save_path = f"results_replicated/{algorithm}/"
            save_file = f"{project}_{repeat}_repetitions.csv"
            save_raw_file = f"{project}_{repeat}_repetitions_raw.csv"
            # Here you would save the metrics to a CSV file at the specified path.
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            # Save raw values (e.g., all accuracy scores from each repetition) to a separate CSV for detailed analysis.
            df_raw = pd.DataFrame({
                'training_time': metrics['training_time'],
                'prediction_time': metrics['prediction_time'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'mcc': metrics['mcc'],
                'cm': [str(cm.tolist()) for cm in metrics['cm']]  # Save the list of confusion matrices as a string in the CSV.
            })
            df_raw.to_csv(os.path.join(save_path, save_raw_file), index=True)
            print(f"Saved raw metrics for {algorithm} on {project} to {os.path.join(save_path, save_raw_file)}")
            
            # Format the metrics as a DataFrame and save to CSV.
            rows['Repeated_Times'].append(repeat)
            rows['Training_Time_Mean'].append(np.mean(metrics['training_time']))
            rows['Prediction_Time_Mean'].append(np.mean(metrics['prediction_time']))
            rows['Accuracy_Mean'].append(np.mean(metrics['accuracy']))
            rows['Accuracy_Median'].append(np.median(metrics['accuracy']))
            rows['Accuracy_Q1'].append(np.percentile(metrics['accuracy'], 25))
            rows['Accuracy_Q3'].append(np.percentile(metrics['accuracy'], 75))
            rows['Precision_Mean'].append(np.mean(metrics['precision']))
            rows['Precision_Median'].append(np.median(metrics['precision']))
            rows['Precision_Q1'].append(np.percentile(metrics['precision'], 25))
            rows['Precision_Q3'].append(np.percentile(metrics['precision'], 75))
            rows['Recall_Mean'].append(np.mean(metrics['recall']))
            rows['Recall_Median'].append(np.median(metrics['recall']))
            rows['Recall_Q1'].append(np.percentile(metrics['recall'], 25))
            rows['Recall_Q3'].append(np.percentile(metrics['recall'], 75))
            rows['F1_Mean'].append(np.mean(metrics['f1']))
            rows['F1_Median'].append(np.median(metrics['f1']))
            rows['F1_Q1'].append(np.percentile(metrics['f1'], 25))
            rows['F1_Q3'].append(np.percentile(metrics['f1'], 75))
            rows['AUC_Mean'].append(np.mean(metrics['auc']))
            rows['AUC_Median'].append(np.median(metrics['auc']))
            rows['AUC_Q1'].append(np.percentile(metrics['auc'], 25))
            rows['AUC_Q3'].append(np.percentile(metrics['auc'], 75))
            rows['MCC_Mean'].append(np.mean(metrics['mcc']))
            rows['MCC_Median'].append(np.median(metrics['mcc']))
            rows['MCC_Q1'].append(np.percentile(metrics['mcc'], 25))
            rows['MCC_Q3'].append(np.percentile(metrics['mcc'], 75))
            rows['CM_Mean'].append(np.mean([np.array(cm) for cm in metrics['cm']], axis=0))

        df_summary = pd.DataFrame({
            'Repeated_Times': rows['Repeated_Times'],
            'Training_Time_Mean': rows['Training_Time_Mean'],
            'Prediction_Time_Mean': rows['Prediction_Time_Mean'],
            'Accuracy_Mean': rows['Accuracy_Mean'],
            'Accuracy_Median': rows['Accuracy_Median'],
            'Accuracy_Q1': rows['Accuracy_Q1'],
            'Accuracy_Q3': rows['Accuracy_Q3'],
            'Precision_Mean': rows['Precision_Mean'],
            'Precision_Median': rows['Precision_Median'],
            'Precision_Q1': rows['Precision_Q1'],
            'Precision_Q3': rows['Precision_Q3'],
            'Recall_Mean': rows['Recall_Mean'],
            'Recall_Median': rows['Recall_Median'],
            'Recall_Q1': rows['Recall_Q1'],
            'Recall_Q3': rows['Recall_Q3'],
            'F1_Mean': rows['F1_Mean'],
            'F1_Median': rows['F1_Median'],
            'F1_Q1': rows['F1_Q1'],
            'F1_Q3': rows['F1_Q3'],
            'AUC_Mean': rows['AUC_Mean'],
            'AUC_Median': rows['AUC_Median'],
            'AUC_Q1': rows['AUC_Q1'],
            'AUC_Q3': rows['AUC_Q3'],
            'MCC_Mean': rows['MCC_Mean'],
            'MCC_Median': rows['MCC_Median'],
            'MCC_Q1': rows['MCC_Q1'],
            'MCC_Q3': rows['MCC_Q3'],
            'CM_Mean': rows['CM_Mean'],
        })
        df_summary.to_csv(os.path.join(save_path, f"{project}_summary.csv"), index=False)
        print(f"Saved summary metrics for {algorithm} on {project} to {os.path.join(save_path, f'{project}_summary.csv')}\n\n")

