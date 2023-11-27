import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
)
from function import calculate_class_weights, calculate_class_weights_2

from tabulate import tabulate
import matplotlib.pyplot as plt

DATASETS = ["dataset.csv", "diabetes.csv"]

CLASSIFIERS = [
    SVC(kernel="linear", random_state=100),
    SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=100,
    )
    # ,SVC(
    #     kernel="rbf",
    #     class_weight="balanced",
    #     random_state=100,
    # ),
]

# rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))
f1_metrics = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))


def choose_class_weight_function(est_idx):
    if est_idx == 1:
        return calculate_class_weights
    else:
        return calculate_class_weights_2


plt.figure(figsize=(15, 5 * len(DATASETS)))
for est_idx, est in enumerate(CLASSIFIERS):
    print("est_idx:", est_idx)
    # plt.figure(figsize=(15, 5 * len(DATASETS)))

    # Pętla po datasetach
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

        # for ds_idx, dataset_filename in enumerate(DATASETS):
        #     plt.figure(figsize=(15, 5))
        #     dataset = pd.read_csv(
        #         dataset_filename,
        #         sep=";",
        #         skiprows=1,
        #     )
        #     X = dataset.iloc[:, [0, 1]].values
        #     y = dataset.iloc[:, -1].values
        #     for est_idx, est in enumerate(CLASSIFIERS):
        #         plt.subplot(1, len(CLASSIFIERS), est_idx + 1)
        for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
            clf = clone(est)

            class_weight_function = choose_class_weight_function(est_idx)
            class_weights = class_weight_function(y[train])

            if est_idx == 1:
                print("Weszlo idx =1 ")
                # if "class_weight" in clf.get_params():
                clf.set_params(class_weight=calculate_class_weights(y[train]))
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            score = balanced_accuracy_score(y[test], y_pred)
            scores[ds_idx, est_idx, fold_idx] = score
            f1 = f1_score(y[test], y_pred)
            f1_metrics[ds_idx, est_idx, fold_idx] = f1

            if X.shape[1] == 2:
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", marker="o")
                plt.title(f"Dataset {ds_idx + 1} - Classifier {est_idx + 1}")

                h = 0.02  # Rozmiar kroku w meshgrid
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
plt.show()


table = tabulate(
    np.mean(scores, axis=-1),
    tablefmt="grid",
    headers=["Srednia", "SVM", "custom_1", "custom_2"],
    showindex=["dataset", "diabetes"],
)

f1_table = tabulate(
    np.mean(f1_metrics, axis=-1),
    tablefmt="grid",
    headers=["f1", "SVM", "custom_1", "custom_2"],
    showindex=["dataset", "diabetes"],
)
print(table)
print("\n", f1_table)


###################################################################################################################################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.base import clone
# from sklearn.metrics import balanced_accuracy_score, f1_score
# from sklearn.model_selection import RepeatedStratifiedKFold

# DATASETS = ["dataset.csv", "diabetes.csv"]

# CLASSIFIERS = [
#     SVC(kernel="linear", random_state=100),
#     # SVC(
#     #     kernel="rbf",
#     #     class_weight="balanced",
#     #     random_state=100,
#     # ),
#     # SVC(
#     #     kernel="rbf",
#     #     class_weight="balanced",
#     #     random_state=100,
#     # ),
# ]

# rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
# scores = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))
# f1_metrics = np.zeros(shape=(len(DATASETS), len(CLASSIFIERS), 2 * 5))


# def choose_class_weight_function(est_idx):
#     if est_idx == 1:
#         return calculate_class_weights
#     else:
#         return calculate_class_weights_2


# # Pętla po klasyfikatorach
# for est_idx, est in enumerate(CLASSIFIERS):
#     plt.figure(figsize=(15, 5 * len(DATASETS)))

#     # Pętla po datasetach
#     for ds_idx, dataset_filename in enumerate(DATASETS):
#         plt.subplot(
#             len(DATASETS), len(CLASSIFIERS), ds_idx * len(CLASSIFIERS) + est_idx + 1
#         )

#         dataset = pd.read_csv(
#             dataset_filename,
#             sep=";",
#             skiprows=1,
#         )
#         X = dataset.iloc[:, [0, 1]].values
#         y = dataset.iloc[:, -1].values

#         clf = clone(est)
#         class_weight_function = choose_class_weight_function(est_idx)
#         class_weights = class_weight_function(y)

#         if "class_weight='balanced'" in clf.get_params():
#             # Jeśli klasyfikator obsługuje class_weight, ustaw go
#             clf.set_params(class_weight=class_weights)
#         clf.fit(X, y)

#         # Wyświetlanie granicy decyzyjnej (tylko dla dwóch cech)
#         if X.shape[1] == 2:
#             plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", marker="o")
#             plt.title(f"Dataset {ds_idx + 1} - Classifier {est_idx + 1}")

#             h = 0.02  # Rozmiar kroku w meshgrid
#             x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#             y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#             xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#             Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)

#             plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
#             plt.xlabel("Feature 1")
#             plt.ylabel("Feature 2")

#     plt.tight_layout()

# plt.show()
