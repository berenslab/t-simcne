#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, redo
from cnexp.plot import add_letters, get_default_metadata
from cnexp.plot.scalebar import add_scalebar_frac


def plot_norms(path, ax, title, rng):
    """Plot the L2 norms of the embedding"""

    # redo.redo_ifchange(path / "intermediates.zip")

    with zipfile.ZipFile(path / "out/intermediates.zip") as zipf:
        with zipf.open("labels.npy") as f:
            labels = np.load(f)

        with zipf.open("embeddings/post.npy") as f:
            ar = np.load(f).astype(float)

    norms = (ar**2).sum(1) ** 0.5
    sc = ax.scatter(
        rng.normal(labels, 0.125),
        norms,
        c=labels,
        alpha=0.5,
        rasterized=True,
    )
    ax.set_title(title)

    # x-axis
    ax.spines.bottom.set_bounds(labels.min(), labels.max())
    if sys.argv[2] == "cifar":
        ax.set_xticks(range(0, 10, 3))
    else:
        ax.set_xticks([labels.min(), labels.max()])
    ax.set_xlabel("class index")

    # y-axis
    ymin, _ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin))
    ax.set_ylabel(r"$\|\mathbf{z}\|$", usetex=True, fontsize="large")

    return sc


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    default = dict(
        euclidean=prefix / names.default_train(),
        cosine=prefix / names.default_train(metric="cosine"),
    )
    # redo.redo_ifchange_slurm(
    #     [d / "default.run" for d in default.values()],
    #     name="norms",
    #     time_str="18:30:00",
    #     partition="gpu-2080ti-preemptable",
    # )
    redo.redo_ifchange(
        [
            stylef,
            inspect.getfile(add_scalebar_frac),
            inspect.getfile(add_letters),
            inspect.getfile(names),
        ]
    )

    rng = np.random.default_rng(511622144)
    with plt.style.context(stylef):
        # maybe rearrange to all be in one line?
        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(2.75, 5.5 / 4),
            sharex=True,
            constrained_layout=True,
        )
        for ax, (k, v) in zip(axs, default.items()):
            plot_norms(v, ax, f"{k.capitalize()}", rng=rng)
        # plot_norms(default["euclidean"], axs[1], titles, rng=rng)

        add_letters(axs)
        # [ax.set(xlabel=None) for ax in axs[:-1, :].flat]
        [ax.set(ylabel=None) for ax in axs[1:]]

    metadata = get_default_metadata()
    metadata[
        "Title"
    ] = "L2 norms of data points in latent space learnt by Euc/cos CLR"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
