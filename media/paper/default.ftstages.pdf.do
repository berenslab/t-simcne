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


def plot_finetune_stages(ft_path, axs, titles):
    """Plot the three stages within ft_path

    Plots the different stages of training for finetuning.  (1) just
    after changing the last layer to 2D, before starting the
    optimization.  (2) after optimizing the last linear layer only.
    (3) after optimizing the entire network.

    """
    assert len(axs) == 3, f"Expected a list with three axes, but got {axs = }"
    assert len(titles) == 3, f"Expected three titles, but got {titles = }"

    final = ft_path
    ft_lin = ft_path.parent
    while not ft_lin.name.startswith("train"):
        ft_lin = ft_lin.parent

    initial = ft_lin.parent
    while not initial.name.startswith("train"):
        initial = initial.parent

    paths = [initial, ft_lin, final]
    redo.redo_ifchange([p / "intermediates.zip" for p in paths])

    with zipfile.ZipFile(ft_lin / "intermediates.zip") as zipf:
        with zipf.open("embeddings/pre.npy") as f:
            pre_emb = np.load(f)

        with zipf.open("labels.npy") as f:
            labels = np.load(f)

    with zipfile.ZipFile(ft_lin / "intermediates.zip") as zipf:
        with zipf.open("embeddings/post.npy") as f:
            ft_emb = np.load(f)
    with zipfile.ZipFile(final / "intermediates.zip") as zipf:
        with zipf.open("embeddings/post.npy") as f:
            final_emb = np.load(f)

    embs = [pre_emb, ft_emb, final_emb]
    scatters = []
    for ax, ar, title in zip(axs, embs, titles):
        sc = ax.scatter(
            ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
        )
        ax.set_title(title)
        add_scalebar_frac(ax)
        scatters.append(sc)

    return scatters


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    default = prefix / names.default_train()
    ft = default / names.finetune()
    redo.redo_ifchange_slurm(
        ft / "default.run",
        name="ftstages",
        partition="gpu-v100-preemptable",
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

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(2.5, 0.75),
            constrained_layout=True,
        )
        titles = ["pre", "ft lin", "final"]
        plot_finetune_stages(ft, axs, titles)

        add_letters(axs)

    metadata = get_default_metadata()
    metadata["Title"] = "Different stages of finetuning for SimCLR to 2D"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
