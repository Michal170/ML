import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel, ttest_ind
import os
import pandas as pd

DATASETS = [
    "glass.csv",
    "diabetes.csv",
    "haberman.csv",
    "vehicle1.csv",
    "yeast4.csv",
    "yeast6.csv",
    "poker-8-9_vs_5.csv",
    "poker-8_vs_6.csv",
]
CLASSIFIERS_names = ["SVM", "SVM_balanced", "SVM_dbscan", "SVM_optics"]

scores = np.load("results/scores.npy")
f1 = np.load("results/f1_result.npy")


table_f1 = tabulate(
    np.mean(f1, axis=-1),
    tablefmt="grid",
    headers=["Metryka f1", "SVM", "SVM_balanced", "SVM_dbscan", "SVM_optics"],
    showindex=DATASETS,
)

table_score = tabulate(
    np.mean(scores, axis=-1),
    tablefmt="grid",
    headers=["Score", "SVM", "SVM_balanced", "SVM_dbscan", "SVM_optics"],
    showindex=DATASETS,
)

data = []
data = []

for i in range(len(CLASSIFIERS_names)):
    headers = []
    headers.append(f"Datasets:")
    for l in range(len(CLASSIFIERS_names)):
        if i == l:
            continue
        else:
            headers.append(f"{CLASSIFIERS_names[i]}:{CLASSIFIERS_names[l]}")
    for k in range(len(DATASETS)):
        row_data = [f"{DATASETS[k]}"]
        count = 0
        for j in range(len(DATASETS)):
            if i == j or count == 3:
                continue
            else:
                count += 1
                result = ttest_rel(f1[k, i, :], f1[k, j, :])
                row_data.extend(
                    [
                        f"{round(result[0],3)} [{result.statistic > 0}] | {round(result[1],3)} [{result.pvalue < 0.05}]",
                    ]
                )
        data.append(row_data)
    table = tabulate(data, headers, tablefmt="grid")
    os.makedirs("results", exist_ok=True)
    output_file_path = os.path.join("results", f"t_test_part_{i}.txt")
    with open(output_file_path, "w") as file:
        file.write(table)
    table_lt = tabulate(data, headers, tablefmt="latex")
    with open(output_file_path, "a") as file:
        file.write("\n\n\n\n\n\n\n\n\n")
        file.write(table_lt)
    data = []

tables = table_f1 + "\n\n" + table_score + "\n\n"
results_directory = "results"
os.makedirs(results_directory, exist_ok=True)


output_file_path = os.path.join(results_directory, "f1_score.txt")
with open(output_file_path, "w") as file:
    file.write(tables)
