#!/usr/bin/env python

import inspect
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, plot, redo
from matplotlib.offsetbox import AnchoredText, AnnotationBbox, OffsetImage
from torchvision.transforms.functional import to_pil_image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def draw_arch(ax, dataset, model, Y, ax_scatter, rng):
    ax.set_axis_off()

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
        r"$\displaystyle\frac{1}"
        r"{1 + ||\mathbf{\mathrm{z}}_i - \mathbf{\mathrm{z}}_j||^2}$"
    )
    txtkwargs = dict(usetex=True, fontsize="large", ha="center", va="center")
    similarity = ax.annotate(txt_sim, (1, 0.5), **txtkwargs)

    z_aprops = aprops.copy()
    rad = -np.pi / 6
    z_aprops["connectionstyle"] = f"arc3,rad={rad}"
    z_aprops["shrinkB"] = 12

    # z_i -> similarity
    txt_zi = r"$\mathbf{\mathrm{z}}_i$"
    z_i = ax.annotate(
        txt_zi,
        similarity.xy,
        (0.75, 1),
        xycoords="axes fraction",
        arrowprops=z_aprops,
        **txtkwargs,
    )

    # z_j -> similarity
    rad *= -1
    z_aprops["connectionstyle"] = f"arc3,rad={rad}"
    txt_zj = r"$\mathbf{\mathrm{z}}_j$"
    z_j = ax.annotate(
        txt_zj,
        similarity.xy,
        (0.75, 0),
        xycoords="axes fraction",
        arrowprops=z_aprops,
        **txtkwargs,
    )

    h_aprops = aprops.copy()

    # h_i -> z_i
    txt_hi = r"$\mathbf{\mathrm{h}}_i$"
    h_i = ax.annotate(
        txt_hi,
        z_i.get_position(),
        (0.5, 1),
        xycoords="axes fraction",
        **txtkwargs,
        arrowprops=h_aprops,
    )
    # h_j -> z_j
    txt_hj = r"$\mathbf{\mathrm{h}}_j$"
    h_j = ax.annotate(
        txt_hj,
        z_j.get_position(),
        (0.5, 0),
        xycoords="axes fraction",
        arrowprops=h_aprops,
        **txtkwargs,
    )

    ix = rng.integers(len(dataset))
    rng.integers(2**32, size=50)
    torch.manual_seed(rng.integers(2**64, dtype="uint"))
    zoom = 0.75
    orig_im, _ = dataset.dataset[ix]
    (t_im1, t_im2), label = dataset[ix]

    # img1 -> h_i
    im1 = OffsetImage(to_pil_image(t_im1), zoom=zoom)
    abox1 = AnnotationBbox(
        im1,
        h_i.get_position(),
        (0.2, 1),
        xycoords="axes fraction",
        box_alignment=(0.5, 1),
        frameon=False,
        arrowprops=aprops,
    )
    ax.add_artist(abox1)

    # img2 -> h_j
    pil_im = to_pil_image(t_im2)
    eprint(pil_im)
    im2 = OffsetImage(pil_im, zoom=zoom)
    abox2 = AnnotationBbox(
        im2,
        h_j.get_position(),
        (0.2, 0),
        xycoords="axes fraction",
        box_alignment=(0.5, 0),
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
    ax.arrow

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
    z_aprops["color"] = "xkcd:light gray"
    z_aprops["linestyle"] = (0, (5, 5))  # dashed
    z_aprops["shrinkA"] = z_aprops["shrinkB"]
    z_aprops["linewidth"] *= 1.5

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
    rad = -0.4
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


def draw_losses(ax, losses):
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")


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
    model = torch.load("seed-3118/model.pt", map_location="cpu")["model"]

    # those might not exist on another computer, so check that the
    # correct embedding is loaded in Y.
    Y = np.load("seed-3118/cifar.npy")
    labels = np.load("seed-3118/labels.npy")

    rng = np.random.default_rng(235997313216654)
    with plt.style.context(stylef):
        fig, axd = plt.subplot_mosaic(
            [["arch", "emb"], ["arch", "loss"]],
            gridspec_kw=dict(width_ratios=[0.75, 0.25]),
            figsize=(5.5, 2),
            constrained_layout=True,
        )

        draw_arch(axd["arch"], dataset, model, Y, axd["emb"], rng=rng)

        ax = axd["emb"]
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            c=labels,
            alpha=0.5,
            # buggy if I rasterize?
            rasterized=True,
        )
        # add_scalebar_frac(ax)
        ax.set_axis_off()
        ax.axis("equal")

        draw_losses(axd["loss"], np.linspace(7.6, 2.0, 1500))
        plot.add_letters(axd.values())
        # plot.add_lettering(axd["arch"], "a")
    metadata = plot.get_default_metadata()
    metadata["Title"] = "Architecture for contrastive visualization"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
