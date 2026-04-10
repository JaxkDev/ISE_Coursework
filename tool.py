######################
##     TOOL CLI     ##
######################

# This file is the main entry point for a PoC (proof-of-concept) for this CLI tool.
# It is designed to have been trained prior and a model saved to disk to load and evaluate.
# But as a PoC, it will train the model from scratch each time and print out prediction based on input.



import os
from argparse import ArgumentParser

# get CLI arguments for which algorithm and project to run, and how many repetitions to do.
get_parser = ArgumentParser(description="Analysis tool for evaluating whether a bug report is a true positive.")
get_parser.add_argument("--algorithm", type=str, default='final', help="The algorithm to analyse (e.g., 'final', 'baseline', 'improved_1', etc.)", choices=['baseline', 'improved_1', 'improved_2', 'improved_3', 'improved_4', 'improved_5', 'final']) #no improved_6
get_parser.add_argument("--project", type=str, default='all', help="The project/dataset to analyse (e.g., 'all', 'pytorch', 'tensorflow', etc.)", choices=['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe', 'all'])
get_parser.add_argument("--file", type=str, required=True, help="The input file to predict from, should be a simple text file with title and body of issue.")
args = get_parser.parse_args()

algorithm_to_run = args.algorithm
project_to_run = args.project
input_file = args.file

if not os.path.exists(input_file):
    print(f"Input file {input_file} does not exist. Please provide a valid input file.")
    exit(1)

print("Loading algorithms... (May take a moment to initialise pytorch)")
from src.algorithms import get_algorithm

with open(input_file, 'r', encoding='utf-8') as file:
    input_data = file.read()

print(f"""
Running prediction tool with:
Algorithm: {algorithm_to_run},
Project: {project_to_run},
Input File: {input_file}
""")

algorithm = get_algorithm(algorithm_to_run)
algorithm.load_dataset(project_to_run)
algorithm.preprocess_data()
algorithm.load_model(fresh=False)  # Load the pre-trained model (if implemented, otherwise it will train from scratch)
metrics = algorithm.train(repetitions=20, seed=51003)  # Train/evaluate the model (or just evaluate if model loading is implemented)

# print(f"\n\nMetrics for training {algorithm_to_run} on {project_to_run} (repeated {len(metrics['accuracy'])} times):\n")
# print(f"Average Training Time:   {np.mean(metrics['training_time']) / 1e9:.8f} seconds")
# print(f"Average Prediction Time: {np.mean(metrics['prediction_time']) / 1e9:.8f} seconds")
# print(f"Average Accuracy:        {np.mean(metrics['accuracy']):.4f} (median: {np.median(metrics['accuracy']):.4f}, Q1: {np.percentile(metrics['accuracy'], 25):.4f}, Q3: {np.percentile(metrics['accuracy'], 75):.4f})")
# print(f"Average Precision:       {np.mean(metrics['precision']):.4f} (median: {np.median(metrics['precision']):.4f}, Q1: {np.percentile(metrics['precision'], 25):.4f}, Q3: {np.percentile(metrics['precision'], 75):.4f})")
# print(f"Average Recall:          {np.mean(metrics['recall']):.4f} (median: {np.median(metrics['recall']):.4f}, Q1: {np.percentile(metrics['recall'], 25):.4f}, Q3: {np.percentile(metrics['recall'], 75):.4f})")
# print(f"Average F1 Score:        {np.mean(metrics['f1']):.4f} (median: {np.median(metrics['f1']):.4f}, Q1: {np.percentile(metrics['f1'], 25):.4f}, Q3: {np.percentile(metrics['f1'], 75):.4f})")
# print(f"Average AUC:             {np.mean(metrics['auc']):.4f} (median: {np.median(metrics['auc']):.4f}, Q1: {np.percentile(metrics['auc'], 25):.4f}, Q3: {np.percentile(metrics['auc'], 75):.4f})")
# print(f"Average MCC:             {np.mean(metrics['mcc']):.4f} (median: {np.median(metrics['mcc']):.4f}, Q1: {np.percentile(metrics['mcc'], 25):.4f}, Q3: {np.percentile(metrics['mcc'], 75):.4f})")
# print(f"Average Confusion Matrix:\n{np.mean([np.array(cm) for cm in metrics['cm']], axis=0)}\n\n")

prediction = algorithm.predict(input_data)
print(f"Prediction: {'Positive (1)' if prediction == 1 else 'Negative (0)'}")
