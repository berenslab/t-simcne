#!/usr/bin/env python

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

        plot.add_letters(fig.get_axes())
    metadata = plot.get_default_metadata()
    metadata["Title"] = "Visualization of pretrained ResNets for " + ", ".join(
        dname_dict.values()
    )
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
