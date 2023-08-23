#!/usr/bin/env python

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import plot, redo
from cnexp.plot.scalebar import add_scalebar_frac


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
            10,
            10,
            figsize=(10, 10),
            squeeze=False,
        )
        for dataset, name in zip(datasets, dname_dict.values()):
            npz = np.load(dataset)
            labels = npz["labels"]
            Y = npz["t-SNE(ResNet152)"].astype(float)
            for idx, ax in enumerate(axxs.flat):
                mask = labels == idx
                ax.scatter(
                    Y[~mask][:, 0],
                    Y[~mask][:, 1],
                    c="xkcd:light grey",
                    alpha=0.5,
                    rasterized=True,
                    zorder=4.5,
                )
                ax.scatter(
                    Y[mask][:, 0],
                    Y[mask][:, 1],
                    c="xkcd:neon green",
                    alpha=0.5,
                    rasterized=True,
                    zorder=5,
                )

                add_scalebar_frac(ax)
                plot.add_lettering(ax, str(idx))

                ax.margins(0)

    metadata = plot.get_default_metadata()
    metadata["Title"] = "Visualization of all classes in CIFAR-100"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
