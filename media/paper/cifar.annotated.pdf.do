#!/usr/bin/env python

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tsimcne import names, redo
from tsimcne.plot import add_letters, get_default_metadata
from tsimcne.plot.scalebar import add_scalebar_frac


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


subclusters = {
    "duplicate cars": (-519, -174, -113, 0),
    "colorful cars": (-438, -244, -150, -50),
    "metallic cars": (-482, -204, -150, 105),
    "firetrucks": (-361, -209, -40, 155),
    "sailboats": (-213, -194, -20, 140),
    "boats (aerial view)": (-242, -244, -150, -75),
    "boats (side view)": (-232, -309, 130, -30),
    "grounded planes": (-143, -242, 220, -97),
    "airborne planes": (-66, -210, 143, -50),
    "ratites": (-13, -85, 90, -105),
    "bird heads": (-22, 5, -85, -115),
    "mounted horses": (-145, 4, -256, 20),
    "dark horses": (-95, -4, -275, 110),
    "bright horses": (-115, 44, -235, 135),
    "horse heads": (-10, 70, -150, 109),
    "deer heads": (44, -8, 195, -252),
    "white dogs": (37, 173, 150, 10),
    "brown dogs": (101, 118, 155, -2),
    "black-and-white pets": (32, 91, 224, -42),
    "black pets": (73, 15, 220, -33),
    "frogs, white background": (185, -65, 108, -115),
    "croaking frogs": (132, 10, 161, -110),
    # "frogs, grass": (147, -14),
    # -415, -175 (alternative, if placed higher
    "red cars": (-410, -175, -190, 150),
}


def annotate_cifar(ax, Y, dataset, arrowprops=None):
    if arrowprops is None:
        arrowprops = dict(
            arrowstyle="-",
            linewidth=plt.rcParams["axes.linewidth"],
            color="xkcd:slate gray",
        )

    rng = np.random.default_rng(56199999)
    n_samples = 3
    knn = NearestNeighbors(n_neighbors=15, n_jobs=8)
    knn.fit(Y)
    for key, (x, y, dx, dy) in subclusters.items():
        # ax.scatter([x], [y], marker="x", c="black")
        img_idx = knn.kneighbors([[x, y]], return_distance=False).squeeze()
        imgs = []
        for ix in rng.choice(img_idx, size=n_samples, replace=False):
            im, lbl = dataset[ix]
            imbox = mpl.offsetbox.OffsetImage(im, zoom=0.5)
            imgs.append(imbox)
        imrow = mpl.offsetbox.HPacker(children=imgs, pad=0, sep=1.5)
        txt = mpl.offsetbox.TextArea(key.capitalize())
        annot = mpl.offsetbox.VPacker(
            children=[txt, imrow], pad=0, sep=1, align="center"
        )
        abox = mpl.offsetbox.AnnotationBbox(
            annot,
            (x, y),
            (x + dx, y + dy),
            arrowprops=arrowprops,
            frameon=False,
            # clip_on=False,
        )
        if key == "frogs, white background":
            abox._arrow_relpos = (0, 1)
        elif key == "croaking frogs":
            abox._arrow_relpos = (0, 1)
        elif key == "metallic cars":
            abox._arrow_relpos = (1, 0.5)
        else:
            pass
        # needs to be removed from the layout, otherwise mpl v3.6 and
        # higher will be confused.
        # https://github.com/matplotlib/matplotlib/issues/24453
        abox.set_in_layout(False)
        ax.add_artist(abox)


def main():
    root = Path("../../experiments/cifar")
    stylef = "../project.mplstyle"

    redo.redo_ifchange(
        [
            root / "dataset.pt",
            stylef,
            inspect.getfile(add_scalebar_frac),
            inspect.getfile(add_letters),
            inspect.getfile(names),
        ]
    )

    # get the underlying dataset without augmentations
    dataset = torch.load(root / "dataset.pt")["full_plain"].dataset

    # those might not exist on another computer, so check that the
    # correct embedding is loaded in Y.
    p = Path("seed-3118")
    Y = np.load(p / "cifar.npy")
    labels = np.load(p / "labels.npy")

    with plt.style.context(stylef):
        fig, ax = plt.subplots(
            figsize=(5.5, 3),
            constrained_layout=True,
        )
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            c=labels,
            alpha=0.5,
            rasterized=True,
        )
        # buggy if I don't have un-rasterized points
        ax.scatter(
            Y[:5][:, 0],
            Y[:5][:, 1],
            c=labels[:5],
            alpha=0.5,
        )
        annotate_cifar(ax, Y, dataset)

        classes = dataset.datasets[0].classes
        cm = plt.get_cmap()
        markers = [
            mpl.lines.Line2D(
                [],
                [],
                label=name,
                color=cm(i),
                ls="",
                marker=mpl.rcParams["scatter.marker"],
                markersize=mpl.rcParams["font.size"],
                # markersize=5,
            )
            for i, name in enumerate(classes)
        ]
        legend = ax.legend(
            handles=markers,
            ncol=2,
            fontsize="medium",
            loc="upper left",
            handletextpad=0.1,
            columnspacing=0.1,
        )
        legend.get_frame().set_linewidth(plt.rcParams["axes.linewidth"])
        add_scalebar_frac(ax)

    metadata = get_default_metadata()
    metadata["Title"] = "Annotated subclusters of cifar10"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
