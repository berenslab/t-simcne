#!/usr/bin/env python

import sys
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from cnexp import plot, redo
from cnexp.plot.scalebar import add_scalebar_frac

# from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def lda_ratio(Z, y):
    mean = np.mean(Z, axis=0)
    n_features = Z.shape[1]

    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    for c in np.unique(y):
        Xc = Z[y == c]
        class_mean = np.mean(Xc, axis=0)
        # within-class variance
        Sw += (Xc - class_mean).T @ (Xc - class_mean)
        mean_diff = (class_mean - mean).reshape(n_features, 1)
        # between-class variance
        Sb += np.sum(y == c) * mean_diff @ mean_diff.T

    return np.sum(np.diag(Sb)) / np.sum(np.diag(Sw))


def main():

    root = Path("../../experiments/")
    # prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"
    k = 100

    dname_dict = dict(
        # cifar="CIFAR-10",
        cifar100="CIFAR-100",  # tiny="Tiny ImageNet"
    )

    datasets = []
    for dataname in dname_dict.keys():
        reps = root.parent / f"stats/{dataname}.tsne.pretrained.resnet.npz"
        redo.redo_ifchange(reps)
        datasets.append(reps)

        # we have this many representations (one less due to the labels)
        n_cols = len(list(np.load(reps).keys())) - 1

    with plt.style.context(stylef):
        fig, axxs = plt.subplots(
            len(datasets),
            n_cols,
            figsize=(5.5, 1.5 * len(datasets)),
            squeeze=False,
        )
        for axs, dataset, name in zip(axxs, datasets, dname_dict.values()):
            axs[0].text(
                0,
                1 / 2,
                name,
                fontsize="x-large",
                ha="center",
                va="center",
                transform=axs[0].transAxes,
                rotation="vertical",
            )
            npz = np.load(dataset)
            labels = npz["labels"]
            keys = [k for k in npz.keys() if k != "labels"]
            for ax, key in zip(axs, keys):

                Y = npz[key].astype(float)
                ax.scatter(
                    Y[:, 0],
                    Y[:, 1],
                    c=labels,
                    alpha=0.5,
                    rasterized=True,
                    zorder=4.5,
                )
                ax.set_title(key)

                add_scalebar_frac(ax)

                X_train, X_test, y_train, y_test = train_test_split(
                    Y, labels, test_size=10_000, random_state=11
                )

                n_classes = 10 if name == "CIFAR-10" else 100

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=5,
                    min_samples=5,
                    cluster_selection_epsilon=1,
                )
                # clusterer = MiniBatchKMeans(n_classes, random_state=10110101)
                preds = clusterer.fit_predict(Y)

                silhouette_score = metrics.silhouette_score(
                    Y, preds, sample_size=10000, random_state=44**5
                )
                ari = metrics.adjusted_rand_score(labels, preds)
                ami = metrics.adjusted_mutual_info_score(labels, preds)
                acctxt = (
                    f"ARI = {ari:.2f}\n"
                    f"AMI = {ami:.2f}\n"
                    f"#clusters = {preds.max()}\n"
                    f"sil. (pred.) = {silhouette_score:.2f}\n"
                )
                eprint(
                    f"{key[-10:]}:\tari = {ari:.2f},"
                    f"\tami = {ami:.2f},\tsil. = {silhouette_score:.2f}"
                    f"\t#cluster = {preds.max()}"
                )

                # ax.set_title(acctxt, loc="right", fontsize="small")
                ax.text(
                    1,
                    1,
                    acctxt,
                    fontsize="small",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    ma="right",
                )

                for cluster_idx in np.unique(preds):
                    Xc = Y[preds == cluster_idx]
                    cmean = Xc.mean(axis=0)
                    ax.scatter(
                        [cmean[0]], [cmean[1]], marker="x", c="black", zorder=5
                    )

                ax.margins(0)

        plot.add_letters(axxs)
    metadata = plot.get_default_metadata()
    metadata["Title"] = "Visualization of pretrained ResNets for " + ", ".join(
        dname_dict.values()
    )
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


superclasses = {
    "aquatic mammals": [4, 30, 55, 72, 95],
    "fish": [1, 32, 67, 73, 91],
    "flowers": [54, 62, 70, 82, 92],
    "food containers": [9, 10, 16, 28, 61],
    "fruit and vegetables": [0, 51, 53, 57, 83],
    "household electrical devices": [22, 39, 40, 86, 87],
    "household furniture": [5, 20, 25, 84, 94],
    "insects": [6, 7, 14, 18, 24],
    "large carnivores": [3, 42, 43, 88, 97],
    "large man-made outdoor things": [12, 17, 37, 68, 76],
    "large natural outdoor scenes": [23, 33, 49, 60, 71],
    "large omnivores and herbivores": [15, 19, 21, 31, 38],
    "medium-sized mammals": [34, 63, 64, 66, 75],
    "non-insect invertebrates": [26, 45, 77, 79, 99],
    "people": [2, 11, 35, 46, 98],
    "reptiles": [27, 29, 44, 78, 93],
    "small mammals": [36, 50, 65, 74, 80],
    "trees": [47, 52, 56, 59, 96],
    "vehicles 1": [8, 13, 48, 58, 90],
    "vehicles 2": [41, 69, 81, 85, 89],
}

label_map = [float("inf")] * 100
[
    label_map.__setitem__(v, i)
    for i, vals in enumerate(superclasses.values())
    for v in vals
]
assert float("inf") not in label_map

superclass_names = list(superclasses.keys())


def fine_to_coarse(label_idx: int) -> int:
    """map the cifar100 fine label to its superclass label"""
    return label_map[label_idx]


if __name__ == "__main__":
    main()
