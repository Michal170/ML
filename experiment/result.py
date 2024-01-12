import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel, ttest_ind
import os
import pandas as pd


DATASETS = [
    # "haberman.csv",
    # # "dataset.csv",
    # "diabetes.csv",
    # "glass.csv",
    # "vehicle1.csv",
    # "poker-8_vs_6.csv",
    # "yeast6.csv",
    # "yeast4.csv",
    "datasets/haberman.csv",
    "datasets/diabetes.csv",
    # "datasets/dataset.csv",
    "datasets/glass.csv",
    "datasets/vehicle1.csv",
    "datasets/poker-8_vs_6.csv",
    "datasets/poker-8-9_vs_5.csv",
    "datasets/yeast6.csv",
    "datasets/yeast4.csv",
]
CLASSIFIERS_names = ["SVM", "SVM_balanced", "SVM_dbscan", "SVM_optics"]

scores = np.load("results/scores.npy")
f1 = np.load("results/f1_result.npy")
print(np.shape(f1))


table_f1 = tabulate(
    np.mean(f1, axis=-1),
    tablefmt="grid",
    headers=["Metryka f1", "SVM", "SVM_balanced", "SVM_dbscan", "SVM_optics"],
    # showindex=["glasses"],
    # showindex=[
    #     "haberman",
    #     "dataset",
    #     "diabetes",
    #     "glasses",
    #     "vehicle1.csv",
    #     "poker-8_vs_6.csv",
    #     "yeast6.csv",
    #     "yeast4.csv",
    # ],
    showindex=DATASETS,
)

table_score = tabulate(
    np.mean(scores, axis=-1),
    tablefmt="grid",
    headers=["Score", "SVM", "SVM_balanced", "SVM_dbscan", "SVM_optics"],
    # showindex=["glasses"],
    showindex=DATASETS,
)


print(table_f1, "\n\n", table_score)

# tables = table_f1 + "\n\n" + table_score


# results_directory = "results"
# os.makedirs(results_directory, exist_ok=True)

# # with open("output.txt", "w") as file:
# #     file.write(tables)

# output_file_path = os.path.join(results_directory, "output.txt")
# with open(output_file_path, "w") as file:
#     file.write(tables)


data = []
# print("\nT-TEST:")
for i in range(len(DATASETS)):
    row_data = [DATASETS[i]]

    result = ttest_rel(scores[i, 0, :], scores[i, 1, :])
    # result = ttest_rel(scores[i, 0, :], scores[i, 1, :])
    result_2 = ttest_rel(scores[i, 0, :], scores[i, 2, :])
    # result_2 = ttest_rel(scores[i, 0, :], scores[i, 2, :])

    row_data.extend(
        [
            f"{round(result[0],3)} | {result.statistic > 0}",
            f"{round(result[1],8)} | {result.pvalue < 0.05}",
            f"{round(result_2[0],3)} | {result_2.statistic > 0}",
            f"{round(result_2[1],8)} | {result_2.pvalue < 0.05}",
        ]
    )

    data.append(row_data)

headers = ["Dataset", "own_1 st", "own_1 p value", "own_2 st", "own_2 p value"]
for classifier in CLASSIFIERS_names[:-1]:
    headers.extend([f"{classifier}"])

table = tabulate(data, headers, tablefmt="grid")

# print(table)

tables = table_f1 + "\n\n" + table_score + "\n\n" + table


results_directory = "results"
os.makedirs(results_directory, exist_ok=True)


output_file_path = os.path.join(results_directory, "output.txt")
with open(output_file_path, "w") as file:
    file.write(tables)
