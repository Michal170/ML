import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets

# Załaduj dane (możesz dostosować to do swojego zestawu danych)
# iris = datasets.load_iris()
# X = iris.data[:, :2]  # Weź tylko pierwsze dwie cechy dla prostoty
X, y = datasets.make_classification(
    n_features=2,  # liczba atrybutów zbioru
    n_samples=300,  # liczba generowanych wzorców
    n_informative=2,  # liczba atrybutów informatywnych, tych które zawierają informacje przydatne dla klasyfikacji
    n_repeated=0,  # liczba atrybutów powtórzonych, czyli zduplikowanych kolumn
    n_redundant=0,  # liczba atrybutów nadmiarowych
    flip_y=0.08,  # poziom szumu
    random_state=100,  # ziarno losowości, pozwala na wygenerowanie dokładnie tego samego zbioru w każdym powtórzeniu
    n_classes=2,  # liczba klas problemu
    weights=[0.8, 0.2],
)

# Ustaw DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=2)
clusters = dbscan.fit_predict(X)
# print(clusters)
unique_colors = np.unique(y)

# Znajdź punkty szumu
noise_points = clusters == -1
noise_samples = np.column_stack((X[noise_points], y[noise_points]))
# print(noise_samples)
# print("\n", "shape:", np.shape(noise_samples))
# print(
#     "unique:",
#     np.bincount(y[noise_points]),
# )

count = np.bincount(y[noise_points])
print(sum(count))
print(count)

dict = {0: sum(count) / count[0], 1: 2 * (sum(count) / count[1])}
print(dict)

# Wyświetl wyniki
plt.figure(figsize=(8, 6))

# Wykres dla przypisanych klastrów
for color in unique_colors:
    mask = (clusters == color) & (~noise_points)
    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        label=f"Cluster {color}",
        cmap="viridis",
        s=50,
        marker="o",
    )

# Wykres dla punktów szumu
plt.scatter(
    X[noise_points, 0],
    X[noise_points, 1],
    label="Noise Points",
    color="gray",
    s=50,
    marker="x",
)

plt.title("DBSCAN Clustering with Noise Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("dbscan.png")
# plt.show()
