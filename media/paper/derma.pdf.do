#!/usr/bin/env python

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, plot, redo


def make_cmap():
    xkcdcolors = [
        "sage",
        "cornflower blue",
        "reddish",
        "light brown",
        "goldenrod",
        "light eggplant",
    ]
    colors = [f"xkcd:{c}" for c in xkcdcolors]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "derma", colors, N=len(colors)
    )
    mpl.colormaps.register(cmap, force=True)
    return cmap


def plot_derma(fig, Y, labels, dataset, rng, n_imgs=(3, 2)):
    tdict = plot.get_lettering_fprops()
    letters = iter("abcdefghijkl")
    cls_ixs, cls_cnts = np.unique(labels, return_counts=True)
    sort_ix = np.argsort(cls_cnts)[::-1]
    classes_indices = cls_ixs[sort_ix]

    # cm = make_cmap()
    cm = plt.get_cmap("tab10", lut=7)
    colors = cm(labels)
    fig1, fig2 = fig.subfigures(1, 2, width_ratios=[1, 3], wspace=0.01)

    ax = fig1.add_subplot()
    ax.scatter(
        Y[:, 0], Y[:, 1], c=colors, marker="o", alpha=0.85, rasterized=True
    )
    ax.text(
        0,
        1,
        next(letters),
        ha="left",
        va="top",
        transform=ax.transAxes,
        **tdict,
    )
    # ax.set_title(next(letters), loc="left", **tdict)
    plot.add_scalebar_frac(ax)

    figs = fig2.subfigures(2, 3, hspace=0.005, wspace=0.005)

    lbl_dict = {
        # "0": "actinic keratoses and\nintraepithelial carcinoma",
        "0": "actinic keratoses",
        "1": "basal cell carcinoma",
        "2": "benign keratosis-like lesions",
        "3": "dermatofibroma",
        "4": "melanoma",
        "5": "melanocytic nevi",
        "6": "vascular lesions",
    }
    for i, fig, ltr in zip(classes_indices, figs.flat, letters):
        cls = lbl_dict[str(i)]
        fig.suptitle(cls)
        mask = labels == i

        fig1, fig2 = fig.subfigures(1, 2)
        ax = fig1.add_subplot()
        ax.scatter(
            Y[~mask, 0],
            Y[~mask, 1],
            c="xkcd:light gray",
            zorder=0.5,
            rasterized=True,
        )
        ax.scatter(Y[mask, 0], Y[mask, 1], c=[cm(i)], rasterized=True)
        # ax.set_title(ltr, loc="left", **tdict)
        ax.text(
            0, 1, ltr, transform=ax.transAxes, ha="left", va="top", **tdict
        )
        ax.axis("equal")
        ax.set_axis_off()

        ncols = n_imgs[1]
        nrows = n_imgs[0]
        axs = fig2.subplots(
            nrows, ncols, gridspec_kw=dict(hspace=0.01, wspace=0.01)
        )
        data_ixs = rng.choice(np.argwhere(mask).squeeze(), size=n_imgs).flat
        for ax, ix in zip(axs.flat, data_ixs):
            ax.imshow(dataset[ix][0])
            ax.set_axis_off()


def main():

    root = Path("../../")
    stylef = "../project.mplstyle"
    fname = "derma-data/derma.npz"
    prefix = root / "experiments/derma"
    # path = prefix / "dl" / names.default_train() / names.finetune()

    redo.redo_ifchange(
        [
            prefix / "dataset.pt",
            stylef,
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    rng = np.random.default_rng(520221)

    dataset = torch.load(prefix / "dataset.pt")["full_plain"].dataset
    npz = np.load(fname)

    with plt.style.context(stylef):
        fig = plt.figure(figsize=(5.5, 2))

        labels = npz["labels"]
        Y = npz["data"]

        plot_derma(fig, Y, labels, dataset, rng)

    metadata = plot.get_default_metadata()
    metadata["Title"] = "DermaMNIST"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
