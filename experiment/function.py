import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
import os


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


def calculate_class_weights_dbscan(X, y):
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(X)
    noise_points = clusters == -1
    count = np.bincount(y[noise_points])
    unique_labels = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

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
        0: sum(((count) / count_0) * ((count) / count_0)),
        1: ((sum(count) / count_1) * (sum(count) / count_1)),
    }

    return dict


def calculate_class_weights_optics(X, y):
    optics = OPTICS(eps=0.5, min_samples=2)
    clusters = optics.fit_predict(X)
    noise_points = clusters == -1

    unique_labels = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

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
        0: sum(((count) / count_0) * ((count) / count_0)),
        1: ((sum(count) / count_1) * (sum(count) / count_1)),
    }

    return dict
