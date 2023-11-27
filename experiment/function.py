import numpy as np


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
