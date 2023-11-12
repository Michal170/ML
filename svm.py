import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import SGDClassifier

# dataset = pd.read_csv("dataset.csv")

X, y = datasets.make_classification(
    n_features=2,  # liczba atrybutów zbioru
    n_samples=4000,  # liczba generowanych wzorców
    n_informative=2,  # liczba atrybutów informatywnych, tych które zawierają informacje przydatne dla klasyfikacji
    n_repeated=0,  # liczba atrybutów powtórzonych, czyli zduplikowanych kolumn
    n_redundant=0,  # liczba atrybutów nadmiarowych
    flip_y=0.08,  # poziom szumu
    random_state=1400,  # ziarno losowości, pozwala na wygenerowanie dokładnie tego samego zbioru w każdym powtórzeniu
    n_classes=2,  # liczba klas problemu
    weights=[0.15, 0.85],
)


# X = dataset.iloc[:, [0, 1]].values
# y = dataset.iloc[:, [2]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# classifier = SVC(kernel="poly", C=1, probability=True)
# classifier.fit(X_train, y_train.ravel())

# classifier_mod = SVC(kernel="poly", C=0.01, probability=True)
# classifier_mod.fit(X_train, y_train.ravel())

clf = SGDClassifier(loss="hinge")
clf.fit(X_train, y_train)


class_weights = {0: 1, 1: 10}

wclf = SGDClassifier(loss="squared_hinge")
# wclf = SGDClassifier(loss="hinge", class_weight=class_weights)
wclf.fit(X_train, y_train)


# y_pred = classifier.predict(X_test)


# cl_probabilities = classifier.predict_proba(X_test)
# predict = np.argmax(cl_probabilities, axis=1)


# cm = confusion_matrix(y_test, y_pred)
# accuracy_score = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(cm)
# print(accuracy_score(y_test, y_pred))
# print("Accuracy score:", accuracy_score)
# print("F1 score:", f1)

# # Plot decision boundaries
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # Real labels
# plot_decision_regions(X_test, y_test.ravel(), clf=classifier, ax=ax[0], legend=2)
# ax[0].set_xlabel("Feature 0")
# ax[0].set_ylabel("Feature 1")
# ax[0].set_title("True Labels")
# ax[0].set_xlim(-4, 4)
# ax[0].set_ylim(-4, 4)

# # Predicted labels
# plot_decision_regions(X_test, y_pred, clf=classifier, ax=ax[1], legend=2)
# ax[1].set_xlabel("Feature 0")
# ax[1].set_ylabel("Feature 1")
# ax[1].set_title(f"Predicted Labels \n F1= {round(f1, 3)}")
# ax[1].set_xlim(-4, 4)
# ax[1].set_ylim(-4, 4)


plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

# plot the decision functions for both classifiers
ax = plt.gca()
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="green",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

disp_mod = DecisionBoundaryDisplay.from_estimator(
    wclf,
    X,
    plot_method="contour",
    colors="red",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

plt.legend(
    [disp.surface_.collections[0], disp_mod.surface_.collections[0]],
    ["green - non weighted", "red - weighted"],
    loc="upper right",
)

# snfig, ax = plt.subplots()
# ax.bar(["F1 Score"], [f1], color=["skyblue"])
# ax.set_ylabel("Score")
# ax.set_title("F1 Score")
# ax.set_ylim([0, 1])

# ax.text(
#     0, f1 + 0.02, f"{f1:.2f}", horizontalalignment="center", verticalalignment="center"
# )

plt.show()

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
