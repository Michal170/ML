import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel, ttest_ind
import os

scores = np.load("results/scores.npy")
f1 = np.load("results/f1_result.npy")


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
