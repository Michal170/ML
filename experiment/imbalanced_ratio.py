import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from tabulate import tabulate
import os


DATASETS_name = [
    "glass.csv",
    "diabetes.csv",
    "haberman.csv",
    "vehicle1.csv",
    "yeast4.csv",
    "yeast6.csv",
    "poker-8-9_vs_5.csv",
    "poker-8_vs_6.csv",
]

DATASETS = [
    "datasets/glass.csv",
    "datasets/diabetes.csv",
    "datasets/haberman.csv",
    "datasets/vehicle1.csv",
    "datasets/yeast4.csv",
    "datasets/yeast6.csv",
    "datasets/poker-8-9_vs_5.csv",
    "datasets/poker-8_vs_6.csv",
]
all_class_distributions = pd.DataFrame(columns=["Dataset", "Class", "Sample Count"])
for ds_idx, dataset_filename in enumerate(DATASETS):
    dataset = pd.read_csv(
        dataset_filename,
        sep=";",
        skiprows=1,
    )

    class_distribution = dataset.iloc[:, -1].value_counts().reset_index()
    class_distribution.columns = ["Class", "Sample Count"]

    class_distribution["Dataset"] = f" {DATASETS_name[ds_idx] }"

    all_class_distributions = pd.concat(
        [all_class_distributions, class_distribution], ignore_index=True
    )


class_ratios = (
    all_class_distributions.groupby("Dataset")["Sample Count"].max()
    / all_class_distributions.groupby("Dataset")["Sample Count"].min()
)

imbalance_table = pd.DataFrame(
    {
        "Dataset": class_ratios.index,
        "Imbalance Ratio": class_ratios.values,
    }
)

os.makedirs("results", exist_ok=True)
output_file_path = os.path.join("results", f"imbalance_ratios.txt")
with open(output_file_path, "w") as file:
    file.write(tabulate(all_class_distributions, headers="keys", tablefmt="grid"))
    file.write("\n\n\n")
    file.write(tabulate(all_class_distributions, headers="keys", tablefmt="latex"))
    file.write("\n\n\n")
    file.write(
        tabulate(
            imbalance_table,
            headers="keys",
            tablefmt="grid",
            colalign=("center", "center", "center"),
        )
    )
    file.write("\n\n\n")
    file.write(
        tabulate(
            imbalance_table,
            headers="keys",
            tablefmt="latex",
            colalign=("center", "center", "center"),
        )
    )
