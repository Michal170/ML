import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def calculate_class_weights(y):
    y = np.array(y).ravel()
    class_counts = np.bincount(y)
    n_classes = len(np.unique(y))
    class_weights = len(y) / (n_classes * class_counts)
    return {
        class_label: class_weights[class_label]
        for class_label in range(len(class_counts))
    }


def calculate_class_weights_2(y):
    y = np.array(y).ravel()
    class_counts = np.bincount(y)
    n_classes = len(np.unique(y))
    class_weights = len(y) / (n_classes * class_counts)
    return {
        class_label: class_weights[class_label]
        for class_label in range(len(class_counts))
    }


def calculate_class_weights_3(X, y):
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(X)
    unique_colors = np.unique(y)
    noise_points = clusters == -1
    # plt.figure()
    # unique_clusters = np.unique(clusters)
    # for cluster_label in unique_clusters:
    #     cluster_points = X[clusters == cluster_label]
    #     if cluster_label == -1:
    #         plt.scatter(
    #             cluster_points[:, 0],
    #             cluster_points[:, 1],
    #             color="gray",
    #             marker="x",
    #             s=50,
    #             label="Noise Points",
    #         )
    #     else:
    #         plt.scatter(
    #             cluster_points[:, 0],
    #             cluster_points[:, 1],
    #             label=f"Cluster {cluster_label}",
    #         )

    # plt.title("DBSCAN Clusters and Noise Points")
    # plt.legend()

    # # Zapisz wykres do pliku
    # plt.savefig("svm_dbscan.png")

    count = np.bincount(y[noise_points])
    try:
        count_0 = count[0]
        if count_0 == 0:
            count_0 = 1
    except IndexError:
        count_0 = 1
    try:
        count_1 = count[1]
        if count_1 == 0:
            count_1 = 1
    except IndexError:
        count_1 = 1

    dict = {
        0: 0.05 * sum(((count) / count_0) * ((count) / count_0)),
        1: ((sum(count) / count_1) * (sum(count) / count_1)),
    }

    return dict
