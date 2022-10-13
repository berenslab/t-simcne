#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, plot, redo


def load(dir, extract_name="embeddings/post.npy"):
    with zipfile.ZipFile(dir / "intermediates.zip") as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    pdict = dict(
        euc=prefix / names.default_train(out_dim=2),
        euc_5k=prefix / names.default_train(out_dim=2, n_epochs=5000),
        # needs to be plotted differently, with Hammer projection
        # cos=prefix / names.default_train(out_dim=3, metric="cosine"),
        ft_cos=prefix
        / names.default_train(metric="cosine")
        / names.finetune(),
        ft_euc=prefix
        / names.default_train(random_state=3118)
        / names.finetune(random_state=3118),
    )
    titles = dict(
        euc="Euclidean",
        euc_5k="Euclidean (5000 epochs)",
        cos="Cosine",
        ft_euc=r"Euclidean $\to$ Euclidean",
        ft_cos=r"Cosine $\to$ Euclidean",
    )
    redo.redo_ifchange_slurm(
        [d / "intermediates.zip" for d in pdict.values()],
        name="gallery",
        partition="gpu-2080ti-preemptable",
        time_str="18:30:00",
    )
    redo.redo_ifchange(
        [
            stylef,
            inspect.getfile(plot.add_scalebar_frac),
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    labels = load(pdict["ft_euc"], "labels.npy")
    anchor = load(pdict["ft_euc"])
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=len(pdict),
            figsize=(5.5, 1.43),
            constrained_layout=True,
        )
        cm = plt.get_cmap()
        for key, ax in zip(pdict.keys(), axs):
            ar = load(pdict[key])
            ar = plot.flip_maybe(ar, anchor=anchor)
            ax.scatter(
                ar[:, 0], ar[:, 1], c=cm(labels), alpha=0.5, rasterized=True
            )
            plot.add_scalebar_frac(ax)
            ax.set_title(titles[key])

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
        legend = fig.legend(
            handles=markers,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(0.715, 0.875),
            handletextpad=0.1,
            columnspacing=0,
            borderaxespad=0,
        )
        legend.get_frame().set_linewidth(plt.rcParams["axes.linewidth"])
        legend.get_frame().set_edgecolor("xkcd:light gray")

        plot.add_letters(axs)

    metadata = plot.get_default_metadata()
    metadata["Title"] = f"Various visualizations of the {sys.argv[2]} dataset"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
