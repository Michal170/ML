import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from function import (
    calculate_class_weights,
    calculate_class_weights_2,
    calculate_class_weights_3,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DATASETS = [
    "datasets/haberman.csv",
    "datasets/dataset.csv",
    "datasets/diabetes.csv",
    "datasets/glass.csv",
    "datasets/vehicle1.csv",
    "datasets/poker-8_vs_6.csv",
]
CLASSIFIERS_names = ["SVM", "own_1", "own_2"]
CLASSIFIERS = [
    SVC(kernel="linear", random_state=100, class_weight="balanced"),
    SVC(
        kernel="linear",
        random_state=100,
    ),
    SVC(
        kernel="linear",
        random_state=100,
    ),
]

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))
f1_metrics = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))


def choose_class_weight_function(est_idx):
    if est_idx == 1:
        return calculate_class_weights
    else:
        return calculate_class_weights_3


plt.figure(figsize=(30, 10 * len(DATASETS)))
for est_idx, est in tqdm(enumerate(CLASSIFIERS), desc="Progress Bar"):
    for ds_idx, dataset_filename in enumerate(DATASETS):
        plt.subplot(
            len(DATASETS), len(CLASSIFIERS), ds_idx * len(CLASSIFIERS) + est_idx + 1
        )

        dataset = pd.read_csv(
            dataset_filename,
            sep=";",
            skiprows=1,
        )
        X = dataset.iloc[:, [0, 1]].values
        y = dataset.iloc[:, -1].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(est)

            if est_idx == 1:
                clf.set_params(class_weight=calculate_class_weights(y[train]))
            elif est_idx == 2:
                clf.set_params(
                    class_weight=calculate_class_weights_3(X[train], y[train])
                )
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            score = balanced_accuracy_score(y[test], y_pred)
            scores[ds_idx, est_idx, fold_idx] = score
            f1 = f1_score(y[test], y_pred)
            f1_metrics[ds_idx, est_idx, fold_idx] = f1

            if X.shape[1] == 2:
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", marker="o")
                plt.title(
                    f"Dataset {DATASETS[ds_idx] } - Classifier {CLASSIFIERS_names[est_idx]}"
                )

                h = 0.02
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
                )
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")

    plt.tight_layout()
results_directory = "results"
os.makedirs(results_directory, exist_ok=True)
plt.savefig(os.path.join(results_directory, "result.png"))

np.save(os.path.join(results_directory, "scores.npy"), scores)

with open(os.path.join(results_directory, "scores.txt"), "w") as outfile:
    outfile.write("# Array shape: {0}\n".format(scores.shape))

    for data_slice in scores:
        np.savetxt(outfile, data_slice, fmt="%-7.2f")

        outfile.write("# New slice\n")
np.save(os.path.join(results_directory, "scores"), scores)
with open(os.path.join(results_directory, "f1_result.txt"), "w") as outfile:
    outfile.write("# Array shape: {0}\n".format(f1_metrics.shape))
    for data_slice in f1_metrics:
        np.savetxt(outfile, data_slice, fmt="%-7.2f")
        outfile.write("# New slice\n")
np.save(os.path.join(results_directory, "f1_result"), f1_metrics)
