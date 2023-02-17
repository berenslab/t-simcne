#!/usr/bin/env python

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, neighbors
from tsimcne import plot, redo
from tsimcne.plot.scalebar import add_scalebar_frac


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():

    root = Path("../../experiments/")
    # prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    dname_dict = dict(
        cifar="CIFAR-10",
        cifar100="CIFAR-100",  # tiny="Tiny ImageNet"
    )

    datasets = []
    for dataname in dname_dict.keys():
        reps = root.parent / f"stats/{dataname}.tsne.pretrained.resnet.npz"
        # redo.redo_ifchange(reps)
        datasets.append(reps)
        rng = np.random.default_rng(2323**5)

        # we have this many representations (one less due to the labels)
        n_cols = len(list(np.load(reps).keys())) - 1

    with plt.style.context(stylef):
        fig = plt.figure(figsize=(5.5, 1.5 * len(datasets) * 2))

        figs = fig.subfigures(2, 1)
        for sfig, dataset, name in zip(figs, datasets, dname_dict.values()):
            axxs = sfig.subplots(2, n_cols // 2)
            sfig.suptitle(name)

            npz = np.load(dataset)
            labels = npz["labels"]
            keys = [k for k in npz.keys() if k != "labels"]
            train, test = model_selection.train_test_split(
                np.arange(labels.shape[0]),
                test_size=10_000,
                random_state=rng.integers(2**32),
                stratify=labels,
            )

            for i, (ax, key) in enumerate(zip(axxs.flat, keys)):

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
                ax.margins(0)

                # knn accuracy
                X_train, X_test, y_train, y_test = (
                    X[train],
                    X[test],
                    labels[train],
                    labels[test],
                )
                knn = neighbors.KNeighborsClassifier(15)
                knn.fit(X_train, y_train)
                acc = knn.score(X_test, y_test)
                acctxt = f"$k$nn = {acc:.0%}"

                # ax.set_title(acctxt, loc="right", fontsize="small")
                ax.text(
                    1,
                    1,
                    acctxt,
                    fontsize="small",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                )

        plot.add_letters(fig.get_axes())
    metadata = plot.get_default_metadata()
    metadata["Title"] = "Visualization of pretrained ResNets for " + ", ".join(
        dname_dict.values()
    )
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
