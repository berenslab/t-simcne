#!/usr/bin/env python

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import redo
from cnexp.plot import get_default_metadata


def cifar_legend(axxs, dataset, rng):
    classes = dataset.datasets[0].classes
    c = len(classes)
    n = 10
    im, lbl = dataset[0]
    im = np.array(im)
    imgs = np.empty((c, n, *im.shape), dtype=im.dtype)
    counts = np.zeros(c, dtype="uint8")
    while any(counts != 10):
        ix = rng.integers(len(dataset))
        im, lbl = dataset[ix]
        if counts[lbl] < 10:
            imgs[lbl, counts[lbl]] = np.array(im)
            counts[lbl] += 1

    cm = plt.get_cmap()
    for axs, imgs_c, class_idx in zip(axxs, imgs, range(c)):
        ax = axs[0]
        ax.set_aspect(1, adjustable="box", anchor="SW")
        radius = 0.1
        eps = 0.05
        dot = mpl.patches.Circle(
            (radius + eps, 0.5),
            radius,
            facecolor=cm(class_idx),
            transform=ax.transAxes,
            edgecolor=None,
        )
        ax.add_artist(dot)
        ax.text(
            2 * radius + 3 * eps,
            0.5,
            classes[class_idx],
            transform=ax.transAxes,
            ha="left",
            va="center",
            # fontsize="xx-large",
        )
        for ax, im in zip(axs[1:], imgs_c):
            ax.imshow(im)
    [ax.set_axis_off() for ax in axxs.flat]


def main():

    root = Path("../../experiments/cifar")
    stylef = "../project.mplstyle"

    redo.redo_ifchange(
        [
            root / "dataset.pt",
            stylef,
        ]
    )

    rng = np.random.default_rng(512)
    # get the underlying dataset without augmentations
    dataset = torch.load(root / "dataset.pt")["full_plain"].dataset

    with plt.style.context(stylef):

        n_samples = 10
        fig, axs = plt.subplots(
            10,
            n_samples + 1,
            figsize=(2.5, 2),
            gridspec_kw=dict(width_ratios=[2.5] + [1] * n_samples),
            constrained_layout=False,
        )
        # fig.suptitle("The CIFAR-10 dataset")
        cifar_legend(axs, dataset, rng=rng)

        fig.subplots_adjust(0, 0, 1, 1, wspace=0.09, hspace=0.09)
    metadata = get_default_metadata()
    metadata["Title"] = "Annotated subclusters of cifar10"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
