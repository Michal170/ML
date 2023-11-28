import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from function import calculate_class_weights, calculate_class_weights_2
import matplotlib.pyplot as plt
from tqdm import tqdm

# DATASETS = ["datasets/haberman.csv"]
DATASETS = ["datasets/haberman.csv", "datasets/dataset.csv", "datasets/diabetes.csv"]

CLASSIFIERS = [
    SVC(kernel="linear", random_state=100),
    SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=100,
    ),
]

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
# rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))
f1_metrics = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))


def choose_class_weight_function(est_idx):
    if est_idx == 1:
        return calculate_class_weights
    else:
        return calculate_class_weights_2


plt.figure(figsize=(15, 5 * len(DATASETS)))
for est_idx, est in tqdm(enumerate(CLASSIFIERS), desc="tqdm() Progress Bar"):
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

        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(est)

            class_weight_function = choose_class_weight_function(est_idx)
            class_weights = class_weight_function(y[train])

            if est_idx == 1:
                clf.set_params(class_weight=calculate_class_weights(y[train]))
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            score = balanced_accuracy_score(y[test], y_pred)
            scores[ds_idx, est_idx, fold_idx] = score
            f1 = f1_score(y[test], y_pred)
            f1_metrics[ds_idx, est_idx, fold_idx] = f1

            if X.shape[1] == 2:
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", marker="o")
                plt.title(
                    f"Dataset {DATASETS[ds_idx] } - Classifier {CLASSIFIERS[est_idx]}"
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

plt.savefig("result.png")
np.save("scores", scores)
np.save("f1_result", f1_metrics)
