"""
Run hypothesis testing, used in the report.
"""

from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
import os
import argparse

# --folder=results_replicated --output=boxplot_replicated.pdf
argparse = argparse.ArgumentParser(description="Run hypothesis testing on F1 scores for ALL datasets across algorithms.")
argparse.add_argument("--folder", type=str, default="results", help="Folder containing the replicated results (default: results)")
args = argparse.parse_args()

FOLDER = args.folder

#check directory exists
if not os.path.exists(FOLDER):
    raise FileNotFoundError(f"Results folder not found: {FOLDER}")

# Map algor name -> file
ALGORITHMS = {
    "baseline":    f"{FOLDER}/baseline/all_50_repetitions_raw.csv",
    "improved_1":  f"{FOLDER}/improved_1/all_50_repetitions_raw.csv",
    "improved_2":  f"{FOLDER}/improved_2/all_50_repetitions_raw.csv",
    "improved_3":  f"{FOLDER}/improved_3/all_50_repetitions_raw.csv",
    "improved_4":  f"{FOLDER}/improved_4/all_50_repetitions_raw.csv",
    "improved_5":  f"{FOLDER}/improved_5/all_50_repetitions_raw.csv",
    #"improved_6":  f"{FOLDER}/improved_6/all_1_repetitions_raw.csv",
}

data = {}
for name, csv_path in ALGORITHMS.items():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    # Find the F1 column here by name
    f1_col = next(col for col in df.columns if col.strip().lower() == "f1")
    f1_values = df[f1_col].values
    data[name] = f1_values
    print(f"Loaded {len(f1_values)} repetitions for '{name}'")


def cliffs_delta(x, y):
    count_better = 0
    count_worse = 0

    for xi in x:
        for yi in y:
            if xi > yi:
                count_better += 1
            elif xi < yi:
                count_worse += 1

    d = (count_better - count_worse) / (len(x) * len(y))
    return d

print()

for improved_name in ["improved_1", "improved_2", "improved_3", "improved_4", "improved_5"]:
    stat, p_value = wilcoxon(data["baseline"], data[improved_name])
    cliff_delta = cliffs_delta(data["baseline"], data[improved_name])

    print(f"Wilcoxon test between 'baseline' and '{improved_name}': statistic={stat:.3f}, p-value={p_value:.3e}")
    print(f"Cliff's delta between 'baseline' and '{improved_name}': {cliff_delta:.3f}\n")

for i in range(1, 5):
    stat, p_value = wilcoxon(data[f"improved_{i}"], data["improved_5"])
    cliff_delta = cliffs_delta(data[f"improved_{i}"], data["improved_5"])
    print(f"Wilcoxon test between 'improved_{i}' and 'improved_5': statistic={stat:.3f}, p-value={p_value:.3e}")
    print(f"Cliff's delta between 'improved_{i}' and 'improved_5': {cliff_delta:.3f}\n")



# Loaded 50 repetitions for 'baseline'
# Loaded 50 repetitions for 'improved_1'
# Loaded 50 repetitions for 'improved_2'
# Loaded 50 repetitions for 'improved_3'
# Loaded 50 repetitions for 'improved_4'
# Loaded 50 repetitions for 'improved_5'

# Wilcoxon test between 'baseline' and 'improved_1': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'baseline' and 'improved_1': -1.000

# Wilcoxon test between 'baseline' and 'improved_2': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'baseline' and 'improved_2': -1.000

# Wilcoxon test between 'baseline' and 'improved_3': statistic=12.000, p-value=1.243e-13
# Cliff's delta between 'baseline' and 'improved_3': -0.903

# Wilcoxon test between 'baseline' and 'improved_4': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'baseline' and 'improved_4': -1.000

# Wilcoxon test between 'baseline' and 'improved_5': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'baseline' and 'improved_5': -1.000

# Wilcoxon test between 'improved_1' and 'improved_5': statistic=1.000, p-value=3.553e-15
# Cliff's delta between 'improved_1' and 'improved_5': -0.966

# Wilcoxon test between 'improved_2' and 'improved_5': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'improved_2' and 'improved_5': -0.968

# Wilcoxon test between 'improved_3' and 'improved_5': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'improved_3' and 'improved_5': -1.000

# Wilcoxon test between 'improved_4' and 'improved_5': statistic=0.000, p-value=1.776e-15
# Cliff's delta between 'improved_4' and 'improved_5': -1.000
