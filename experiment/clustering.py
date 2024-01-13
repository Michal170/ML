import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
import os
import pandas as pd


DATASETS = [
    "datasets/glass.csv",
    "datasets/diabetes.csv",
    "datasets/haberman.csv",
    "datasets/vehicle1.csv",
    "datasets/yeast4.csv",
    "datasets/yeast6.csv",
    "datasets/poker-8-9_vs_5.csv",
    "datasets/poker-8_vs_6.csv",
]
DATASETS_name = [
    "glass.csv",
    "diabetes.csv",
    "haberman.csv",
    "vehicle1.csv",
    "yeast4.csv",
    "yeast6.csv",
    "poker-8-9_vs_5.csv",
    "poker-8_vs_6.csv",
]


def paint_cluster(name, filename):
    num_datasets = len(DATASETS)
    num_rows = (num_datasets + 1) // 2
    num_cols = 2

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for ds_idx, dataset_filename in enumerate(DATASETS):
        dataset = pd.read_csv(
            dataset_filename,
            sep=";",
            skiprows=1,
        )
        X = dataset.iloc[:, [0, 1]].values
        y = dataset.iloc[:, -1].values

        methods = name(eps=0.5, min_samples=2)
        clusters = methods.fit_predict(X)
        noise_points = clusters == -1
        count = np.bincount(y[noise_points])
        unique_labels = set(clusters)
        colors = [
            plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
        ]

        row_idx = ds_idx // num_cols
        col_idx = ds_idx % num_cols

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0.5, 0.5, 0.5, 1]
                marker = "x"
            else:
                marker = "o"
            class_member_mask = clusters == k

            xy = X[class_member_mask]
            axs[row_idx, col_idx].plot(
                xy[:, 0],
                xy[:, 1],
                marker,
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        axs[row_idx, col_idx].set_title(f"{DATASETS_name[ds_idx]}")

    if num_datasets % 2 != 0:
        fig.delaxes(axs[-1, -1])

    plt.tight_layout()
    results_directory = "results"
    os.makedirs(results_directory, exist_ok=True)
    plt.savefig(os.path.join(results_directory, f"{filename}.png"), dpi=300)


if __name__ == "__main__":
    paint_cluster(DBSCAN, "dbscan")
    paint_cluster(OPTICS, "optics")
