#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, plot, redo


def main():

    root = Path("../../")
    stylef = "../project.mplstyle"

    # redo.redo_ifchange_slurm(
    #     [path / "intermediates.zip" for path in paths],
    #     name=[f"seed-{seed}" for seed in seeds],
    #     partition="gpu-2080ti-preemptable",
    #     time_str="18:30:00",
    # )
    tsnes = [
        root / f"stats/cifar{c}.{r}.npz"
        for c in ["", "100"]
        for r in ["pixelspace", "tsne.simclr"]
    ]
    titles = [
        f"CIFAR-{c} in {r} space"
        for c in [10, 100]
        for r in ["pixel", "SimCLR"]
    ]
    redo.redo_ifchange(
        tsnes
        + [
            stylef,
            inspect.getfile(plot.add_scalebar_frac),
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    labels = np.load(tsnes[0])["labels"]
    anchor = np.load(tsnes[0])["tsne"]

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            ncols=len(tsnes),
            figsize=(5.5, 1.85),
            constrained_layout=True,
        )
        for npzf, ax, t in zip(tsnes, axs, titles):
            npz = np.load(npzf)
            ar = npz["tsne"]
            ar = plot.flip_maybe(ar, anchor=anchor)
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            plot.add_scalebar_frac(ax)
            ax.set_title(t)

        plot.add_letters(axs)

    metadata = plot.get_default_metadata()
    metadata[
        "Title"
    ] = "t-SNE visualizations of CIFAR-10/100 with SimCLR and pixel-based"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
