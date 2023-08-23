#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, redo
from cnexp.plot import add_letters, get_default_metadata, get_lettering_fprops
from cnexp.plot.scalebar import add_scalebar_frac
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def load(dir, extract_name="embeddings/post.npy"):
    with zipfile.ZipFile(dir / "intermediates.zip") as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


def main():

    root = Path("../../experiments/")
    path = (
        root
        / "derma/dl/model/sgd/lrcos/infonce/train"
        / "ftmodel:freeze=1:change=lastlin/sgd/lrcos:n_epochs=50:warmup_epochs=0"
        / "train/ftmodel:freeze=0/sgd:lr=0.00012/lrcos:n_epochs=450/train/intermediates.zip"
    )
    stylef = "../project.mplstyle"

    redo.redo_ifchange_slurm(
        path,
        name="derma",
        partition="gpu-2080ti-preemptable",
        time_str="18:30:00",
    )
    redo.redo_ifchange(
        [
            stylef,
            inspect.getfile(add_scalebar_frac),
            inspect.getfile(add_letters),
            inspect.getfile(names),
        ]
    )

    with zipfile.ZipFile(path) as zf:
        with zf.open("embeddings/post.npy") as f:
            Y = np.load(f)
        with zf.open("labels.npy") as f:
            labels = np.load(f)

    dataset = torch.load(root / "derma/dataset.pt")["full_plain"]
    rng = np.random.default_rng(398185418)

    lbl_dict = {
        "0": "actinic keratoses and\nintraepithelial carcinoma",
        "1": "basal cell carcinoma",
        "2": "benign keratosis-like lesions",
        "3": "dermatofibroma",
        "4": "melanoma",
        "5": "melanocytic nevi",
        "6": "vascular lesions",
    }
    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(5.5, 5.5),
            constrained_layout=True,
        )
        cm = plt.get_cmap(lut=10)  # len(lbl_dict)

        ax1 = axs[0, 1]

        markers = [
            mpl.lines.Line2D(
                [],
                [],
                label=val,
                color=cm(int(key)),
                ls="",
                marker=mpl.rcParams["scatter.marker"],
                markersize=mpl.rcParams["font.size"],
            )
            for key, val in lbl_dict.items()
        ]
        legend = axs[0, 0].legend(handles=markers, loc="center")
        legend.get_frame().set_linewidth(mpl.rcParams["axes.linewidth"])

        for label, ax in zip(range(len(lbl_dict)), axs.flat[2:]):
            Ym = Y[labels == label]
            Yo = Y[labels != label]
            n = (labels == label).sum()
            ax.set_title(lbl_dict[str(label)])
            frac = n / labels.shape[0]
            ax.set_title(f"{frac:.1%}", fontsize="small", loc="right")
            ax1.scatter(
                Ym[:, 0],
                Ym[:, 1],
                c=[cm(label)],
                marker="o",
                alpha=0.85,
                zorder=4 - frac,
                rasterized=True,
            )

            # for some weird reason rasterizing the scatter plots
            # causes the image annotations to be misplaced (they end
            # up in the legend axes for some inexplicable reason).
            # Probably a bug, but I don't have the time to dig into it
            # currently.
            sc1 = ax.scatter(
                Yo[:, 0],
                Yo[:, 1],
                c="xkcd:light gray",
                marker="o",
                # rasterized=True,
            )
            sc2 = ax.scatter(
                Ym[:, 0],
                Ym[:, 1],
                c=[cm(label)],
                marker="o",  # , rasterized=True
            )
            # xmin, ymin = Y.argmin(axis=0)
            # xmax, ymax = Y.argmax(axis=0)
            # ax.scatter(
            #     [Y[xmin, 0], Y[ymin, 0], Y[xmax, 0], Y[ymax, 0]],
            #     [Y[xmin, 1], Y[ymin, 1], Y[xmax, 1], Y[ymax, 1]],
            #     zorder=0.9,
            #     color="white",
            # )

            # whether data lies in the upper cluster.  Limit has been
            # determined via visual inspection.
            is_upper = Y[:, 1] > 25
            label_ix_upper = np.argwhere(
                (labels == label) & is_upper
            ).squeeze()
            n_samples = min(5, label_ix_upper.shape[0])
            sample_ix_upper = rng.choice(
                label_ix_upper, n_samples, replace=False
            )
            xslots = np.linspace(
                Y[:, 0].min(), Y[:, 0].max(), n_samples  # , endpoint=False
            )
            _, ymargin = ax.margins()
            ymax = Y[:, 1].max() * (1 + ymargin)
            ymin = Y[:, 1].min() * (1 + ymargin)

            aboxes = []
            for x, six in zip(xslots, sample_ix_upper):
                im, _ = dataset.dataset[six]
                imbox = OffsetImage(im, zoom=0.5)
                xy = (x, ymax)
                ab = AnnotationBbox(
                    imbox,
                    xy,
                    xycoords="data",
                    box_alignment=(0.5, 0),
                    frameon=False,
                )
                ax.add_artist(ab)
                aboxes.append(ab)

            label_ix_lower = np.argwhere(
                (labels == label) & ~is_upper
            ).squeeze()
            sample_ix_lower = rng.choice(
                label_ix_lower, n_samples, replace=False
            )
            ax.update_datalim(Y)
            for (x, six) in zip(xslots, sample_ix_lower):
                im, _ = dataset.dataset[six]
                imbox = OffsetImage(im, zoom=0.5)
                xy = (x, ymin)
                ab = AnnotationBbox(
                    imbox,
                    xy,
                    xycoords="data",
                    box_alignment=(0.5, 1),
                    frameon=False,
                )
                ax.add_artist(ab)
                aboxes.append(ab)

            # include the annotation box in the datalim
            # https://stackoverflow.com/questions/11545062/
            # matplotlib-autoscale-axes-to-include-annotations
            fig.canvas.draw()
            for ab in aboxes:
                bbox = ab.get_window_extent(fig.canvas.get_renderer())
                bbox_data = bbox.transformed(ax.transData.inverted())
                corners = bbox_data.corners()
                ax.update_datalim(corners)
            ax.autoscale_view()

        [(ax.axis("equal"), ax.set_axis_off()) for ax in axs.flat]
        add_scalebar_frac(ax1)
        add_letters(axs.flat[1:])
        axs[0, 2].set_title("b", fontdict=get_lettering_fprops(), loc="left")

    metadata = get_default_metadata()
    metadata["Title"] = (
        "DermaMNIST dataset plotted by class, "
        "with samples from upper and lower cluster displayed"
    )
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
