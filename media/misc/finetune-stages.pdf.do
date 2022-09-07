#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnexp import names, redo
from cnexp.plot import add_lettering, get_default_metadata
from cnexp.plot.scalebar import add_scalebar_frac


def main():

    prefix = Path("../../experiments/cifar/dl/")
    stylef = "../project.mplstyle"

    # train_kwargs = dict(device=1)
    train_kwargs = dict()
    default = prefix / names.default_train(
        train_kwargs=train_kwargs,
    )
    last = default / names.finetune(train_kwargs=train_kwargs)
    ft_lastlin = last.parent
    while not ft_lastlin.name.startswith("train"):
        ft_lastlin = ft_lastlin.parent

    path_dict = dict(
        ft_lin=ft_lastlin, last=last
    )  # simclr=default, last=last)
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
            default / "intermediates.zip",
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

    with plt.style.context(stylef):
        fig, axd = plt.subplot_mosaic(
            "ab\ncd",
            figsize=(3, 3),
            constrained_layout=True,
        )

        for (ltr, ax), (key, ar) in zip(axd.items(), path_dict.items()):
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            ax.set_title(key)
            add_scalebar_frac(ax)
            add_lettering(ax, ltr)

        # ax = axd["b"]
        # add_lettering(ax, "b")
        # with zipfile.ZipFile(ft_lastlin / "intermediates.zip") as zipf:
        #     with zipf.open("embeddings/pre.npy") as f:
        #         ar = np.load(f)

        # ax.scatter(ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True)
        # ax.set_title("before last lin. opt")
        # add_scalebar_frac(ax)

        ax = axd["c"]
        add_lettering(ax, "c")
        with zipfile.ZipFile(default / "intermediates.zip") as zipf:
            with zipf.open("embeddings/post.npy") as f:
                ar = np.load(f)
        norms = (ar**2).sum(axis=1) ** 0.5
        rng = np.random.default_rng(567812309)
        ax.scatter(
            rng.normal(labels, 0.15),
            norms,
            c=labels,
            alpha=0.5,
            rasterized=True,
        )
        ax.set_title(f"simclr {ar.shape[1]}d vector norms")
        ax.set_xlabel("class index")
        ax.set_ylabel("$||x||_2$")

        ax = axd["d"]
        add_lettering(ax, "d")
        ax.plot(losses, c="xkcd:dark gray")
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        # ax.set_xticks(list(range(0, losses.index[-1], 500)))
        ax.set_xmargin(0)

    metadata = get_default_metadata()
    metadata["Title"] = "Different stages of finetuning for SimCLR to 2D"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
