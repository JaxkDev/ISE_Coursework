"""
Box plot of F1 scores for the 'ALL' datasets across all algorithms.
Output: f1_boxplot.pdf (NOT SCREENSHOT!!)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# --folder=results_replicated --output=boxplot_replicated.pdf
argparse = argparse.ArgumentParser(description="Generate box plot of F1 scores for ALL datasets across algorithms.")
argparse.add_argument("--folder", type=str, default="results", help="Folder containing the replicated results (default: results)")
argparse.add_argument("--output", type=str, default="graphs/f1_boxplot.pdf", help="Output PDF file name (default: graphs/f1_boxplot.pdf)")
args = argparse.parse_args()

FOLDER = args.folder
OUTPUT_FILE = args.output

#check directory exists
if not os.path.exists(FOLDER):
    raise FileNotFoundError(f"Results folder not found: {FOLDER}")

# check output file name ends with .pdf
if not OUTPUT_FILE.lower().endswith(".pdf"):
    raise ValueError(f"Output file name must end with .pdf: {OUTPUT_FILE}")

# if output file has directories, check they exist
output_dir = os.path.dirname(OUTPUT_FILE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

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


baseline_mean = np.mean(data["baseline"])
improved5_mean = np.mean(data["improved_5"])
print(f"Baseline mean F1: {baseline_mean:.3f}")
print(f"Improved_5 mean F1: {improved5_mean:.3f}")


#PLOT

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

labels = list(ALGORITHMS.keys())
boxes_data = [data[name] for name in labels]
labels = [label.title() for label in labels]

# Create box plot styles
box = dict(facecolor="lightgrey", edgecolor="black", linewidth=1.2)
median = dict(color="blue", linewidth=1.5, solid_capstyle="round")
#mean = dict(marker="D", markerfacecolor="black", markersize=6, markeredgecolor="black", markeredgewidth=0.8)
flier = dict(marker="o", markerfacecolor="black", markersize=4, alpha=0.5)

bp = ax.boxplot(
    boxes_data,
    tick_labels=labels,
    patch_artist=True,
    #showmeans=True,
    #meanprops=mean,
    medianprops=median,
    flierprops=flier,
    boxprops=box,
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)

ax.axhline(y=baseline_mean, color="red", linestyle="--", linewidth=1, label=f"Baseline mean ({baseline_mean:.3f})")
ax.axhline(y=improved5_mean, color="green", linestyle="--", linewidth=1, label=f"Improved_5 mean ({improved5_mean:.3f})")
ax.set_xlabel("Algorithm", fontweight="bold", fontsize=12)
ax.set_ylabel("F1 Score", fontweight="bold", fontsize=12)
ax.set_title("F1 Score across ALL Datasets", fontweight="bold", fontsize=14, pad=10)
ax.legend(loc="lower right", frameon=True, fontsize=10) #Use lower right to avoid overlapping with boxes... damn window resize made me waste time here
ax.grid(axis="y", alpha=0.5)

# Add buffer to y-axis limits
# Otherwise the whiskers just coincide with the plot border :/
y_min = min(np.min(box) for box in boxes_data) - 0.02
y_max = max(np.max(box) for box in boxes_data) + 0.02
ax.set_ylim(y_min, y_max)


plt.tight_layout()

## NOT A SCREENSHOT, PROPER PDF EXPORT REQUIRED..
plt.savefig(OUTPUT_FILE, format="pdf", bbox_inches="tight")
print(f"Box plot saved to {OUTPUT_FILE}")

plt.show()