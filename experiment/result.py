import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel, ttest_ind
import os
import pandas as pd

DATASETS = [
    "datasets/haberman.csv",
    "datasets/dataset.csv",
    "datasets/diabetes.csv",
    "datasets/glass.csv",
    "datasets/vehicle1.csv",
    "datasets/poker-8_vs_6.csv",
]
CLASSIFIERS_names = ["SVM", "own_1", "own_2"]

scores = np.load("results/scores.npy")
f1 = np.load("results/f1_result.npy")
print(np.shape(f1))


table_f1 = tabulate(
    np.mean(f1, axis=-1),
    tablefmt="grid",
    headers=["f1", "SVM", "SVM_weights", "SVM_dbscan"],
    # showindex=["glasses"],
    showindex=[
        "haberman",
        "dataset",
        "diabetes",
        "glasses",
        "vehicle1.csv",
        "poker-8_vs_6.csv",
    ],
)

table_score = tabulate(
    np.mean(scores, axis=-1),
    tablefmt="grid",
    headers=["score", "SVM", "SVM_weights", "SVM_dbscan"],
    # showindex=["glasses"],
    showindex=[
        "haberman",
        "dataset",
        "diabetes",
        "glasses",
        "vehicle1.csv",
        "poker-8_vs_6.csv",
    ],
)


print(table_f1, "\n\n", table_score)

tables = table_f1 + "\n\n" + table_score


results_directory = "results"
os.makedirs(results_directory, exist_ok=True)

# with open("output.txt", "w") as file:
#     file.write(tables)

output_file_path = os.path.join(results_directory, "output.txt")
with open(output_file_path, "w") as file:
    file.write(tables)


for i in range(len(DATASETS) - 1):
    # print(i, j)
    result = ttest_rel(scores[i, 0, :], scores[i, 1, :])
    result_2 = ttest_rel(scores[i, 0, :], scores[i, 2, :])
    print(
        result.statistic > 0,
        result.pvalue < 0.05,
        result_2.statistic > 0,
        result_2.pvalue < 0.05,
    )
    # print(result.pvalue < 0.05)
# result = ttest_rel(scores[1, 1, :], scores[1, 2, :])
# print(result[0], result[1])

data = []

for i in range(len(DATASETS) - 1):
    row_data = [DATASETS[i]]

    result = ttest_rel(scores[i, 0, :], scores[i, 1, :])
    result_2 = ttest_rel(scores[i, 0, :], scores[i, 2, :])

    row_data.extend(
        [
            f"{round(result[0],3)} | {result.statistic > 0}",
            f"{round(result[1],3)} | {result.pvalue < 0.05}",
            f"{round(result_2[0],3)} | {result_2.statistic > 0}",
            f"{round(result_2[1],3)} | {result_2.pvalue < 0.05}",
        ]
    )

    data.append(row_data)

# Nagłówki dla kolumn
headers = ["Dataset", "own_1 st", "own_1 p value", "own_2 st", "own_2 p value"]
for classifier in CLASSIFIERS_names[:-1]:
    headers.extend([f"{classifier}"])

# Utworzenie tabeli z danymi
table = tabulate(data, headers, tablefmt="grid")

# Wyświetlenie tabeli
print(table)
