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
    "haberman.csv",
    "diabetes.csv",
    # "dataset.csv",
    "glass.csv",
    "vehicle1.csv",
    "poker-8_vs_6.csv",
    "poker-8-9_vs_5.csv",
    "yeast6.csv",
    "yeast4.csv",
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
print("\nT-TEST:")
# for i in range(len(DATASETS)):
#     row_data = [DATASETS[i]]

#     # result = ttest_rel(scores[i, 0, :], scores[i, 1, :])
#     # result_2 = ttest_rel(scores[i, 0, :], scores[i, 2, :])
#     # result_3 = ttest_rel(scores[i, 0, :], scores[i, 3, :])
#     result = ttest_rel(f1[i, 0, :], f1[i, 1, :])
#     result_2 = ttest_rel(f1[i, 0, :], f1[i, 2, :])
#     result_3 = ttest_rel(f1[i, 0, :], f1[i, 3, :])

#     row_data.extend(
#         [
#             f"{round(result[0],3)} | {result.statistic > 0}",
#             f"{round(result[1],8)} | {result.pvalue < 0.05}",
#             f"{round(result_2[0],3)} | {result_2.statistic > 0}",
#             f"{round(result_2[1],8)} | {result_2.pvalue < 0.05}",
#             f"{round(result_3[0],3)} | {result_3.statistic > 0}",
#             f"{round(result_3[1],8)} | {result_3.pvalue < 0.05}",
#         ]
#     )

#     data.append(row_data)

# headers = [
#     "Dataset \\ SVC with:",
#     "balanced statistic",
#     "balanced p_value",
#     "dbscan statistic",
#     "dbscan p_value",
#     "optics statistic",
#     "optics p_value",
# ]
# for classifier in CLASSIFIERS_names[:-1]:
#     headers.extend([f"{classifier}"])

# table = tabulate(data, headers, tablefmt="grid")

# print(table)

# tables = table_f1 + "\n\n" + table_score + "\n\n" + table


# results_directory = "results"
# os.makedirs(results_directory, exist_ok=True)


# output_file_path = os.path.join(results_directory, "output.txt")
# with open(output_file_path, "w") as file:
#     file.write(tables)
headers = [
    "Clf \\ Dataset:",
    "haberman[statistic]",
    "haberman[p_value]",
    "diabetes[statistic]",
    "diabetes[p_value]",
    "glass[statistic]",
    "glass[p_value]",
    "vehicle1[statistic]",
    "vehicle1[p_value]",
    "poker-8_vs_6[statistic]",
    "poker-8_vs_6[p_value]",
    "poker-8-9_vs_5[statistic]",
    "poker-8-9_vs_5[p_value]",
    "yeast6[statistic]",
    "yeast6[p_value]",
    "yeast4[statistic]",
    "yeast4[p_value]",
]
for j in range(4):
    for i in range(len(CLASSIFIERS_names)):
        row_data = [f"{CLASSIFIERS_names[j]}:{CLASSIFIERS_names[i]}"]
        if i == j:
            continue
        else:
            for k in range(len(DATASETS)):
                result = ttest_rel(f1[k, j, :], f1[k, i, :])

                row_data.extend(
                    [
                        f"{round(result[0],3)} | {result.statistic > 0} ",
                        f"{round(result[1],8)} | {result.pvalue < 0.05} ",
                    ]
                )

            data.append(row_data)

table = tabulate(data, headers, tablefmt="grid")

print(table)

tables = table_f1 + "\n\n" + table_score + "\n\n" + table


results_directory = "results"
os.makedirs(results_directory, exist_ok=True)


output_file_path = os.path.join(results_directory, "output.txt")
with open(output_file_path, "w") as file:
    file.write(tables)
