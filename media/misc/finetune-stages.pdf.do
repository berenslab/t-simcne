#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnexp import names, redo
from cnexp.plot import add_lettering
from cnexp.plot.scalebar import add_scalebar_frac

if __name__ == "__main__":

    prefix = Path("../../experiments/cifar/dl/")
    stylef = "../project.mplstyle"

    default = prefix / names.default_train()
    last = default / names.finetune()
    ft_lastlin = last.parent
    while not ft_lastlin.name.startswith("train"):
        ft_lastlin = ft_lastlin.parent

    path_dict = dict(simclr=default, ft_lin=ft_lastlin)  # , last=last)
    fnames = [
        path / f
        for path in path_dict.values()
        for f in ["intermediates.zip", "losses.csv"]
    ]
    redo.redo_ifchange_slurm(
        ft_lastlin / "default.run",
        name="50 epoch ft",
        partition="gpu-v100-preemptable",
        time_str="00:30:00",
    )
    redo.redo_ifchange(
        fnames
        + [
            default / "losses.csv",
            stylef,
            inspect.getfile(add_scalebar_frac),
            inspect.getfile(add_lettering),
            inspect.getfile(names),
        ]
    )

    for key, p in path_dict.items():
        with zipfile.ZipFile(p / "intermediates.zip") as zipf:
            with zipf.open("embeddings/post.npy") as f:
                ar = np.load(f)
                path_dict[key] = ar

            with zipf.open("labels.npy") as f:
                labels = np.load(f)

    loss_fnames = [d / "losses.csv" for d in [default, ft_lastlin]]  # , last]]
    losses = pd.concat(
        (pd.read_csv(f)["mean"] for f in loss_fnames), ignore_index=True
    )

    blocksize = 1.25
    with plt.style.context(stylef):
        fig, axd = plt.subplot_mosaic(
            "ab\ncd",
            figsize=(len(path_dict) * blocksize, 2 * blocksize + 0.05),
            constrained_layout=True,
        )

        for (ltr, ax), (key, ar) in zip(axd.items(), path_dict.items()):
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            ax.set_title(key)
            add_scalebar_frac(ax)
            add_lettering(ax, ltr)

        ax = axd["c"]
        add_lettering(ax, "c")
        with zipfile.ZipFile(ft_lastlin / "intermediates.zip") as zipf:
            with zipf.open("embeddings/pre.npy") as f:
                ar = np.load(f)

        ax.scatter(ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True)
        ax.set_title("before last lin. opt")
        add_scalebar_frac(ax)

        ax = axd["d"]
        add_lettering(ax, "d")
        ax.plot(losses, c="xkcd:dark gray")
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        ax.set_xticks(list(range(0, losses.index[-1], 500)))
        ax.set_xmargin(0)

    metadata = dict(
        Author="Jan Niklas Böhm",
        Title="Different stages of finetuning for SimCLR to 2D",
    )
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)
