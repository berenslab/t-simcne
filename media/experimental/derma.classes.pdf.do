#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, redo
from cnexp.plot import add_letters, get_default_metadata, get_lettering_fprops
from cnexp.plot.scalebar import add_scalebar_frac


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
        cm = plt.get_cmap()

        colors = cm(labels)
        axs[0, 1].scatter(
            Y[:, 0], Y[:, 1], c=colors, s=1, alpha=0.85, ec=None, marker="o"
        )
        # axs[0, 1].set_title("DermaMNIST\n")

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
        legend.get_frame().set_linewidth(0.4)
        axs[0, 0].set_axis_off()

        for label, ax in zip(range(8), axs.flat[2:]):
            Ym = Y[labels == label]
            Yo = Y[labels != label]
            n = (labels == label).sum()
            ax.scatter(Yo[:, 0], Yo[:, 1], c="xkcd:light gray", marker="o")
            ax.scatter(Ym[:, 0], Ym[:, 1], c=[cm(label)], marker="o")
            add_scalebar_frac(ax)
            ax.set_title(lbl_dict[str(label)])
            ax.set_title(
                f"{n / labels.shape[0]:.1%}", fontsize="small", loc="right"
            )

        add_letters(axs.flat[1:])
        axs[0, 2].set_title("b", fontdict=get_lettering_fprops(), loc="left")

    metadata = get_default_metadata()
    metadata["Title"] = "DermaMNIST dataset plotted by class"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
