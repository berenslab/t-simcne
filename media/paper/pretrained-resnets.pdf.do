#!/usr/bin/env python

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import plot, redo
from cnexp.plot.scalebar import add_scalebar_frac
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():

    root = Path("../../experiments/")
    # prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"
    rng = np.random.default_rng(511622144)

    dname_dict = dict(
        cifar="CIFAR-10",
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
            figsize=(5.5, 1.15 * len(datasets)),
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
            train, test = train_test_split(
                np.arange(labels.shape[0]),
                test_size=10_000,
                random_state=rng.integers(2**32),
                stratify=labels,
            )
            eprint(train.shape, test.shape)
            for ax, key in zip(axs, keys):

                X = npz[key].astype(float)
                cm = plt.get_cmap(lut=labels.max() + 1)
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    c=labels,
                    alpha=0.5,
                    rasterized=True,
                    zorder=4.5,
                    cmap=cm,
                )
                ax.set_title(key)

                add_scalebar_frac(ax)

                X_train, X_test, y_train, y_test = (
                    X[train],
                    X[test],
                    labels[train],
                    labels[test],
                )

                test_accs = []
                train_accs = []
                for run in range(1):
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty="none",
                            solver="saga",
                            n_jobs=-1,
                            tol=1e-3,
                            random_state=rng.integers(2**32),
                        ),
                    )
                    clf.fit(X_train, y_train)

                    acc = clf.score(X_test, y_test)
                    test_accs.append(acc)
                    train_accs.append(clf.score(X_train, y_train))

                accs = np.array(test_accs)
                train_accs = np.array(train_accs)
                # acctxt = f"acc = {accs.mean() * 100:.0f}±{accs.std():.0%}"
                acctxt = f"acc = {accs.mean():.0%}"
                eprint(
                    f"{name}\t{key[-10:]}:\t"
                    f"train {train_accs.mean():.0%}\t"  # ±{train_accs.std():.0%},\t"
                    f"test {accs.mean():.0%}"  # ±{accs.std():.0%}"
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

                levels = [i - 0.5 for i in range(labels.max() + 2)]
                dbd = DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X,
                    grid_resolution=1000,
                    eps=0,
                    ax=ax,
                    alpha=0.4,
                    levels=levels,
                    cmap=cm,
                )
                for c in dbd.surface_.collections:
                    c.set_rasterized(True)

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
