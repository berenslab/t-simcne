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


def plot_norms(path, axs, titles, rng):
    """Plot the three stages within ft_path

    Plots the different stages of training for finetuning.  (1) just
    after changing the last layer to 2D, before starting the
    optimization.  (2) after optimizing the last linear layer only.
    (3) after optimizing the entire network.

    """
    assert len(axs) == len(titles)

    # redo.redo_ifchange(path / "intermediates.zip")

    with zipfile.ZipFile(path / "out/intermediates.zip") as zipf:
        with zipf.open("labels.npy") as f:
            labels = np.load(f)

        with zipf.open("embeddings/post.npy") as f:
            Z = np.load(f).astype(float)
        with zipf.open("backbone_embeddings/post.npy") as f:
            H = np.load(f).astype(float)

    scatters = []
    for ax, ar, title in zip(axs, [H, Z], titles):
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
        ax.set_ylabel("$||x||$")

        scatters.append(sc)

    return scatters


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
            ncols=4,
            figsize=(5.5, 5.5 / 4),
            sharex=True,
            constrained_layout=True,
        )
        titles = ["H", "Z"]
        for ax, (k, v) in zip(axs.reshape(2, 2), default.items()):
            plot_norms(
                v, ax, [f"{k.capitalize()} ${t}$" for t in titles], rng=rng
            )
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
