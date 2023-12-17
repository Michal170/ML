import numpy as np
from sklearn.cluster import DBSCAN


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
    dbscan = DBSCAN(eps=0.3, min_samples=2)
    clusters = dbscan.fit_predict(X)

    noise_points = clusters == -1
    # noise_samples = np.column_stack((X[noise_points], y[noise_points]))

    count = np.bincount(y[noise_points])
    # print(sum(count))
    # print(count)

    dict = {0: sum(count) / count[0], 1: 2 * (sum(count) / count[1])}
    return dict
