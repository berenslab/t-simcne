#!/usr/bin/env python

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, redo
from cnexp.plot import get_default_metadata


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def cifar_dups(ax, Y, dataset, labels, arrowprops=None):
    if arrowprops is None:
        arrowprops = dict(
            arrowstyle="-",
            linewidth=plt.rcParams["axes.linewidth"],
            color="xkcd:slate gray",
        )

    mask = (Y[:, 0] < -510) & (Y[:, 1] > -210)
    ext_mask = (Y[:, 0] < -500) & (Y[:, 1] > -215)
    imgs = [dataset[ix][0] for ix, b in enumerate(mask) if b]
    Ym = Y[mask]
    eprint(Ym.shape)
    for xy, im in zip(Ym, imgs):
        # ax.scatter([x], [y], marker="x", c="black")
        imbox = mpl.offsetbox.OffsetImage(im, zoom=0.55)
        # txt = mpl.offsetbox.TextArea(key.capitalize())
        # annot = mpl.offsetbox.VPacker(
        #     children=[txt, imrow], pad=0, sep=2, align="center"
        # )
        abox = mpl.offsetbox.AnnotationBbox(
            imbox,
            xy,
            arrowprops=arrowprops,
            frameon=False,
            # clip_on=False,
        )
        ax.add_artist(abox)

    Y1 = Y[ext_mask]
    cm = plt.get_cmap(lut=10)
    clrs = cm(labels[ext_mask])
    ax.scatter(*Y1.T, c=clrs, s=5, rasterized=True)
    ax.update_datalim(Ym)


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
        fig, ax = plt.subplots(figsize=(2.75, 2.75))
        cifar_dups(ax, Y, dataset, labels)
        # ax.axis("equal")
        ax.set_axis_off()

    metadata = get_default_metadata()
    metadata["Title"] = "Annotated subclusters of cifar10"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
