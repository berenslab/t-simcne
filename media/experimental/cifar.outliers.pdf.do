#!/usr/bin/env python

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, plot, redo

outlier_indices = [36218, 48261, 41914, 53557, 58590, 6330, 33796, 49425]


def annotate_cifar(ax, Y, dataset, arrowprops=None):
    if arrowprops is None:
        arrowprops = dict(
            arrowstyle="-",
            linewidth=plt.rcParams["axes.linewidth"],
            color="xkcd:slate gray",
        )
    txtkwargs = plot.get_lettering_fprops()
    locs = Y[outlier_indices]
    imgs = np.array([np.array(dataset[ix][0]) for ix in outlier_indices])

    upper = locs[:, 1] > np.median(locs[:, 1])
    sort_up = np.argsort(locs[:, 0][upper])
    sort_lo = np.argsort(locs[:, 0][~upper])
    locs_s = np.vstack((locs[upper][sort_up], locs[~upper][sort_lo]))
    imgs_s = np.vstack((imgs[upper][sort_up], imgs[~upper][sort_lo]))

    xs = np.linspace(0, 1, len(outlier_indices) // 2, endpoint=False)
    xys = [(x, y) for y in [1, 0] for x in xs]

    letters = "abcdefghijkl"
    for loc, im, xy, ltr in zip(locs_s, imgs_s, xys, letters):
        # ax.scatter([x], [y], marker="x", c="black")
        imbox = mpl.offsetbox.OffsetImage(im, zoom=1)
        txt = mpl.offsetbox.TextArea(ltr, textprops=txtkwargs)
        annot = mpl.offsetbox.VPacker(
            children=[txt, imbox], pad=0, sep=2, align="left"
        )
        abox = mpl.offsetbox.AnnotationBbox(
            annot,
            loc,
            xy,
            boxcoords="axes fraction",
            box_alignment=(0, xy[1]),
            arrowprops=arrowprops,
            frameon=False,
            # clip_on=False,
        )
        ax.add_artist(abox)


def main():

    root = Path("../../experiments/cifar")
    stylef = "../project.mplstyle"

    redo.redo_ifchange(
        [
            root / "dataset.pt",
            stylef,
            inspect.getfile(names),
        ]
    )

    # get the underlying dataset without augmentations
    dataset = torch.load(root / "dataset.pt")["full_plain"].dataset

    # those might not exist on another computer, so check that the
    # correct embedding is loaded in Y.
    p = Path("../paper/seed-3118")
    Y = np.load(p / "cifar.npy")
    labels = np.load(p / "labels.npy")

    with plt.style.context(stylef):
        fig, ax = plt.subplots(
            figsize=(2.75, 2.75),
            # constrained_layout=False,
        )
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            c=labels,
            alpha=0.5,
            # buggy if I rasterize
            # rasterized=True,
        )
        Ym = Y[outlier_indices]
        c = labels[outlier_indices]
        ax.scatter(Ym[:, 0], Ym[:, 1], c=c, s=30)
        annotate_cifar(ax, Y, dataset)
        ax.axis("equal")
        ax.set_axis_off()

    metadata = plot.get_default_metadata()
    metadata["Title"] = "Annotated subclusters of cifar10"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
