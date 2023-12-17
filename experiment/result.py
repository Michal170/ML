import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel, ttest_ind

scores = np.load("scores.npy")
f1 = np.load("f1_result.npy")


table_f1 = tabulate(
    np.mean(f1, axis=-1),
    tablefmt="grid",
    headers=["f1", "SVM", "SVM_weights", "SVM_dbscan"],
    showindex=[
        "haberman",
        "dataset",
        "diabetes",
    ],
)

table_score = tabulate(
    np.mean(scores, axis=-1),
    tablefmt="grid",
    headers=["score", "SVM", "SVM_weights", "SVM_dbscan"],
    showindex=[
        "haberman",
        "dataset",
        "diabetes",
    ],
)

print(table_f1, "\n\n", table_score)

tables = table_f1 + "\n\n" + table_score

with open("output.txt", "w") as file:
    file.write(tables)
