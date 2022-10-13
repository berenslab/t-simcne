#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, plot, redo


def main():

    root = Path("../../")
    stylef = "../project.mplstyle"
    fname = "derma-data/derma.npz"
    prefix = root / "experiments/derma"
    # path = prefix / "dl" / names.default_train() / names.finetune()
    dataset = torch.load(prefix / "dataset.pt")["full_plain"].dataset

    redo.redo_ifchange(
        [
            root / fname,
            stylef,
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    npz = np.load(root / fname)

    with plt.style.context(stylef):
        fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.8))
        labels = npz["labels"]
        Y = npz["data"]
        cm = plt.get_cmap("tab10", lut=10)
        colors = cm(labels)

        ax = axs[0]
        ax.scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.85, rasterized=True)
        plot.add_scalebar_frac(ax)
        # ax.set_title("")

        ax = axs[1]
        Y = npz["tsne"]
        ax.scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.85, rasterized=True)
        plot.add_scalebar_frac(ax)
        ax.set_title("t-SNE")

        ax = axs[2]
        Y = npz["pca"]
        ax.scatter(Y[:, 0], Y[:, 1], c=colors, alpha=0.85, rasterized=True)
        plot.add_scalebar_frac(ax)
        ax.set_title("PCA")

    metadata = plot.get_default_metadata()
    metadata[
        "Title"
    ] = "Comparison between visualization algorithms for DermaMNIST"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
