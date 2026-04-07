# Dataset processing and statistics.

import os
import pandas as pd

DATASETS = [
    "caffe",
    "incubator-mxnet",
    "keras",
    "pytorch",
    "tensorflow",
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../dataset/"

def print_dataset_details():
    total_data = []
    for dataset in DATASETS:
        print(f"{dataset}:")
        with open(f"{DATA_DIR}{dataset}.csv", "r", encoding="utf-8") as f:
            data = pd.read_csv(f).values.tolist()
            num_instances = len(data) - 1  # Exclude header
            class_counts = {}
            for line in data:
                label = line[-2]  # Assuming label is the second-to-last column
                class_counts[label] = class_counts.get(label, 0) + 1

        print(f"  Number of instances: {num_instances}")
        print(f"  Number of classes: {len(class_counts)}")
        print("  Class balance:")
        for label, count in class_counts.items():
            print(f"    {label}: {count}")

        with open(f"{DATA_DIR}{dataset}_details.csv", "w", encoding="utf-8") as f:
            f.write("Class,Count\n")
            for label, count in class_counts.items():
                f.write(f"{label},{count}\n")

        total_data.append((dataset, num_instances, len(class_counts), class_counts))

    print("\nSummary:")
    for dataset, num_instances, num_classes, class_counts in total_data:
        class_distribution = {label: ((count/num_instances) * 100) for label, count in class_counts.items()}
        print(f"{dataset}: {num_instances} instances, {num_classes} classes, class balance: {class_counts}, class balance (normalized): {list(class_distribution.values())[0]:.2f}%")
    print("\nTotal:")
    total_instances = sum([data[1] for data in total_data])
    total_classes = sum([data[2] for data in total_data])
    print(f"  Total instances: {total_instances}")
    print(f"  Total classes: {total_classes}")
    print("  Overall class balance:")
    overall_class_counts = {}
    for _, _, _, class_counts in total_data:
        for label, count in class_counts.items():
            overall_class_counts[label] = overall_class_counts.get(label, 0) + count
    for label, count in overall_class_counts.items():
        print(f"    {label}: {count} / {total_instances} ({(count/total_instances) * 100:.2f}%)")


def preprocess_datasets():
    for project in DATASETS + ['all']:
        pd_all = pd.DataFrame()

        if project == 'all':
            # If 'all', read and concatenate all project files
            for proj in DATASETS:
                path = f'{proj}.csv'
                pd_temp = pd.read_csv(DATA_DIR + path)
                pd_all = pd.concat([pd_all, pd_temp], ignore_index=True)
                pd_all = pd_all.sample(frac=1, random_state=51003)  # Shuffle
        else:
            path = f'{project}.csv'
            pd_all = pd.read_csv(DATA_DIR + path)
            pd_all = pd_all.sample(frac=1, random_state=51003)  # Shuffle

        # Merge Title and Body into a single column; if Body is NaN, use Title only
        pd_all['Title+Body'] = pd_all.apply(
            lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
            axis=1
        )

        # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
        pd_all = pd_all.rename(columns={
            "Unnamed: 0": "id",
            "class": "sentiment",
            "Title+Body": "text"
        })
        pd_all.to_csv(DATA_DIR + project + '_processed.csv', index=False, columns=["id", "Number", "sentiment", "text"])
        print(f"Processed {project} dataset saved to {project}_processed.csv")


if __name__ == "__main__":
    print_dataset_details()
    preprocess_datasets()
