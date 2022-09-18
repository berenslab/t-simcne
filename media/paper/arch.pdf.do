#!/usr/bin/env python

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cnexp import names, plot, redo
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon, Rectangle
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def draw_arch(ax, dataset, model, Y, ax_scatter, rng):
    ax.set_axis_off()

    upper_row = 0.875
    lower_row = 0.125

    aprops = dict(
        arrowstyle="-|>",
        color="xkcd:dark gray",
        linewidth=plt.rcParams["axes.linewidth"],
        relpos=(1, 0.5),
        shrinkB=7,
    )

    # panel a is drawn from the right to the left

    # loss/similarity
    txt_sim = (
        r"\small$\displaystyle\frac{1}"
        r"{1 + \|\mathrm{\mathbf{z}}_i - \mathrm{\mathbf{z}}_j\|^2}$"
    )
    txtkwargs = dict(
        usetex=True,
        horizontalalignment="center",
        verticalalignment="center",
    )
    similarity = ax.annotate(txt_sim, (0.9, 0.5), **txtkwargs)

    z_aprops = aprops.copy()
    rad = -np.pi / 8
    z_aprops["connectionstyle"] = f"arc3,rad={rad}"
    z_aprops["shrinkB"] = 12

    # z_i -> similarity
    txt_zi = r"\small$\mathrm{\mathbf{z}}_i$"
    z_i = ax.annotate(
        txt_zi,
        similarity.xy,
        (0.65, upper_row),
        xycoords="axes fraction",
        arrowprops=z_aprops,
        **txtkwargs,
    )

    # z_j -> similarity
    rad *= -1
    z_aprops["connectionstyle"] = f"arc3,rad={rad}"
    txt_zj = r"\small$\mathrm{\mathbf{z}}_j$"
    z_j = ax.annotate(
        txt_zj,
        similarity.xy,
        (0.65, lower_row),
        xycoords="axes fraction",
        arrowprops=z_aprops,
        **txtkwargs,
    )

    h_aprops = aprops.copy()

    # h_i -> z_i
    txt_hi = r"\small$\mathrm{\mathbf{h}}_i$"
    h_i = ax.annotate(
        txt_hi,
        z_i.get_position(),
        (0.5, upper_row),
        xycoords="axes fraction",
        **txtkwargs,
        arrowprops=h_aprops,
    )

    # h_j -> z_j
    txt_hj = r"\small$\mathrm{\mathbf{h}}_j$"
    h_j = ax.annotate(
        txt_hj,
        z_j.get_position(),
        (0.5, lower_row),
        xycoords="axes fraction",
        arrowprops=h_aprops,
        **txtkwargs,
    )

    torch.manual_seed(rng.integers(2**64, dtype="uint"))
    ix = rng.integers(len(dataset))
    zoom = 0.75
    orig_im, _ = dataset.dataset[ix]
    (t_im1, t_im2), label = dataset[ix]
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    invnorm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]
            ),
            transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
        ]
    )
    t_im1 = invnorm(t_im1)
    t_im2 = invnorm(t_im2)

    # img1 -> h_i
    im1 = OffsetImage(to_pil_image(t_im1), zoom=zoom)
    abox1 = AnnotationBbox(
        im1,
        h_i.get_position(),
        (0.2, upper_row),
        xycoords="axes fraction",
        # box_alignment=(0.5, 1),
        frameon=False,
        arrowprops=aprops,
    )
    ax.add_artist(abox1)

    # img2 -> h_j
    im2 = OffsetImage(to_pil_image(t_im2), zoom=zoom)
    abox2 = AnnotationBbox(
        im2,
        h_j.get_position(),
        (0.2, lower_row),
        xycoords="axes fraction",
        # box_alignment=(0.5, 0),
        frameon=False,
        arrowprops=aprops,
    )
    ax.add_artist(abox2)

    im = mpl.offsetbox.OffsetImage(orig_im, zoom=zoom)
    abox = mpl.offsetbox.AnnotationBbox(
        im,
        (0, 0.5),
        xycoords="axes fraction",
        box_alignment=(0, 0.5),
        frameon=False,
        arrowprops=aprops,
    )
    ax.add_artist(abox)

    augprops = aprops.copy()
    augprops["connectionstyle"] = f"angle,angleA=90,angleB=180"
    augprops["shrinkA"] = 15
    augprops["shrinkB"] = 15
    ax.annotate(
        "",
        (0.2, upper_row),
        (0.05, 0.5),
        xycoords="axes fraction",
        arrowprops=augprops,
    )
    ax.annotate(
        "",
        (0.2, lower_row),
        (0.05, 0.5),
        xycoords="axes fraction",
        arrowprops=augprops,
    )
    t = txtkwargs.copy()
    t["usetex"] = False
    t["fontsize"] = "small"
    t["transform"] = ax.transAxes
    t["rotation_mode"] = "anchor"
    t["horizontalalignment"] = "left"
    ax.text(
        0,
        0,
        "data aug-\nmentation",
        va="bottom",
        **t,
    )
    ax.text(
        0,
        1,
        "data aug-\nmentation",
        va="top",
        **t,
    )
    ax_scatter.scatter(
        [Y[ix, 0]], [Y[ix, 1]], c="black", marker="o", s=3, zorder=4
    )

    # model.eval()
    # with torch.no_grad():
    #     batch = torch.stack([t_im1, t_im2])
    #     feats, bb_feats = model(batch)

    # eprint(feats)
    # for a in feats:
    #     a = a.numpy()
    #     ax_scatter.scatter(
    #         [a[0]], [a[1]], c="black", marker="x", s=15, zorder=4
    #     )

    z_aprops = aprops.copy()
    z_aprops["color"] = "xkcd:gray"
    z_aprops["linestyle"] = (0, (5, 5))  # dashed
    z_aprops["shrinkA"] = z_aprops["shrinkB"]
    z_aprops["shrinkB"] = 1
    z_aprops["linewidth"] *= 1.5
    rad = -0.2
    z_aprops["connectionstyle"] = f"arc3,rad={rad}"

    # connect z_i to scatter plot
    annot1 = plt.Annotation(
        "",
        Y[ix, :],
        z_i.get_position(),
        xycoords=ax_scatter.transData,
        textcoords=ax.transAxes,
        # textcoords="axes fraction",
        arrowprops=z_aprops,
    )
    ax_scatter.add_artist(annot1)

    # connect z_j to scatter plot
    rad = 0.1
    z_aprops["connectionstyle"] = f"arc3,rad={rad}"
    annot2 = plt.Annotation(
        "",
        Y[ix, :],
        z_j.get_position(),
        xycoords=ax_scatter.transData,
        textcoords=ax.transAxes,
        # textcoords="axes fraction",
        arrowprops=z_aprops,
    )
    ax_scatter.add_artist(annot2)

    bkwargs = dict(
        edgecolor="xkcd:slate gray",
        facecolor="xkcd:light gray",
        linewidth=plt.rcParams["axes.linewidth"],
    )
    # resnet polygon
    xy = np.array([[0.285, 0.2], [0.285, 0.8], [0.5, 0.65], [0.5, 0.35]])
    resnet = Polygon(xy, closed=True, **bkwargs)
    t = txtkwargs.copy()
    t["usetex"] = False
    ax.text(*xy.mean(axis=0), "ResNet", **t)
    ax.add_artist(resnet)

    ph_props = aprops.copy()
    ph_props["shrinkB"] = 1
    ph_props["shrinkA"] = 5

    t = txtkwargs.copy()
    t.update(
        color="xkcd:slate gray",
        va="bottom",
        ha="center",
        fontsize="medium",
        usetex=False,
    )

    ax.text(
        0.5,
        0.66,
        "512",
        **t,
    )

    # projection head polygon
    layerwidth = 0.015
    dx = 0.075 - layerwidth / 2
    x = 0.5 + dx
    ax.annotate("", (x, 0.5), (0.5 - layerwidth, 0.5), arrowprops=ph_props)
    hidden_layer = Rectangle((x, 0.25), layerwidth, 0.5)
    ax.text(
        x + layerwidth / 2,
        0.75,
        "1024",
        **t,
    )
    x += dx
    r3height = 0.1
    out_layer = Rectangle((x, 0.5 - r3height / 2), layerwidth, r3height)
    ax.annotate("", (x, 0.5), (x - dx, 0.5), arrowprops=ph_props)
    ax.text(
        x + layerwidth / 2,
        0.55,
        "2",
        **t,
    )

    pc = PatchCollection([hidden_layer, out_layer], **bkwargs)
    t = txtkwargs.copy()
    t["usetex"] = False
    ax.text(
        x - dx,
        0.225,
        "projection head",
        transform=ax.transAxes,
        **t,
        va="top",
    )
    ax.add_collection(pc)


def draw_losses(ax, losses):
    loss = losses["mean"]
    ax.plot(loss)
    ax.set_xlabel("epoch", labelpad=1)
    # label = ax.set_xlabel("epoch", labelpad=-4)
    # label.set_horizontalalignment("right")
    # ax.xaxis.set_label_coords(-0.01, -0.03)
    ax.set_ylabel("loss")
    ax.tick_params("both", labelsize="small")
    ax.spines.left.set_bounds(2.5, 7.5)
    ax.set_yticks([2.5, 5, 7.5])
    ax.spines.bottom.set_bounds(0, 1500)
    ax.set_xticks(range(0, 1501, 500))
    ax.set_aspect(1500 / 5)
    txtkwargs = dict(
        fontsize="small",
        multialignment="center",
        verticalalignment="top",
        horizontalalignment="center",
    )
    ax.annotate(
        "Training\nin 128D",
        (500, 1),
        xycoords=("data", "axes fraction"),
        **txtkwargs,
    )

    ax.axvspan(
        1000,
        1050,
        # ymax=0.75,
        linewidth=plt.rcParams["axes.linewidth"],
        color="xkcd:light gray",
        zorder=1,
    )
    ax.annotate(
        "Finetuning\nin 2D",
        (1050 + 10, 1),
        xycoords=("data", "axes fraction"),
        ha="left",
        **txtkwargs,
    )

    # txtkwargs["fontsize"] = "x-small"
    aprops = dict(
        # arrowstyle="-[,widthB=0.5",
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.5",
        color="xkcd:dark gray",
        linewidth=plt.rcParams["axes.linewidth"],
        relpos=(1, 0.5),
        shrinkB=4,
    )

    ax.annotate(
        "Linear layer\nfinetuning",
        (1025, loss[1025]),
        (500, 5),
        arrowprops=aprops,
        va="center",
        **txtkwargs,
    )


def main():

    root = Path("../../experiments/cifar")
    stylef = "../project.mplstyle"

    redo.redo_ifchange(
        [
            root / "dataset.pt",
            stylef,
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    dataset = torch.load(root / "dataset.pt")["train_contrastive"]
    # model = torch.load("seed-3118/model.pt", map_location="cpu")["model"]
    model = None  # phony, since it's currently unused.

    # those might not exist on another computer, so check that the
    # correct embedding is loaded in Y.
    Y = np.load("seed-3118/cifar.npy")
    labels = np.load("seed-3118/labels.npy")
    losses = pd.read_csv("seed-3118/losses.csv")

    # gives a blue car with greyscale, flipping, and cropping
    # transformation.
    rng = np.random.default_rng(7961039361578664541)
    with plt.style.context(stylef):
        fig, axd = plt.subplot_mosaic(
            [["arch", "emb", "loss"]],
            gridspec_kw=dict(width_ratios=[3, 2.2, 1], hspace=0.05),
            figsize=(5.5, 1.2),
            constrained_layout=False,
        )

        draw_arch(axd["arch"], dataset, model, Y, axd["emb"], rng=rng)

        ax = axd["emb"]
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            c=labels,
            alpha=0.5,
            rasterized=True,
        )
        ax.text(0.33, 1, "CIFAR-10", transform=ax.transAxes, va="top")
        # add_scalebar_frac(ax)
        ax.set_axis_off()
        ax.margins(0)
        ax.axis("equal")

        draw_losses(axd["loss"], losses)
        fig.subplots_adjust(0.0, 0, 0.99, 1)

    metadata = plot.get_default_metadata()
    metadata["Title"] = "Architecture for contrastive visualization"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
