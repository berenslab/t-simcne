#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnexp import names, redo
from cnexp.plot import add_letters, get_default_metadata
from cnexp.plot.scalebar import add_scalebar_frac


def plot_norms(
    ax, labels, ar, rng, set_xlabel=True, set_ylabel=True, title=None
):
    ar = ar.astype(float)
    norms = (ar**2).sum(1) ** 0.5

    ax.scatter(
        rng.normal(labels, 0.125), norms, c=labels, alpha=0.5, rasterized=True
    )
    ax.set_xticks([labels.min(), labels.max()])
    ax.spines.bottom.set_bounds(labels.min(), labels.max())
    ax.spines.left.set_bounds(0, norms.max())

    if set_xlabel:
        ax.set_xlabel("class idx")
    if set_ylabel:
        ax.set_ylabel("||x||")
    if title is not None:
        ax.set_title(title)


def plot_one(path, axs, title, knn=True, loss=True, norm_ylabel=True):

    rng = np.random.default_rng(55**2 - 1)

    final = path
    first = final.parent
    while any(p.startswith("train") for p in first.parts):
        first = first.parent

    paths = [path]
    knnfiles = [p / "knn/score.txt" for p in paths]
    lossfiles = [p / "losses.csv" for p in paths]
    deps = [p / "intermediates.zip" for p in paths]
    if knn:
        deps += knnfiles
    if loss:
        deps += lossfiles
    redo.redo_ifchange(deps)

    knn_it = iter(float(f.read_text()) for f in knnfiles)
    loss_it = iter(pd.read_csv(f)["mean"].tail(1).item() for f in lossfiles)

    with zipfile.ZipFile(path / "intermediates.zip") as zipf:
        with zipf.open("embeddings/post.npy") as f:
            ar = np.load(f)

        with zipf.open("labels.npy") as f:
            labels = np.load(f)

    ax1, ax2 = axs
    ax1.scatter(ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True)
    ax1.set_title(title)
    add_scalebar_frac(ax1)

    t = ""
    if knn:
        f = next(knn_it)
        t += f"knn = {f:.1%}"
    if loss:
        f = next(loss_it)
        t += "\n" if t != "" else ""
        t += f"loss = {f:.1f}"
    if t != "":
        # ax1.set_title(t, loc="right", fontsize="x-small", ma="right")
        ax1.annotate(
            text=t,
            xy=(1, 1),
            xycoords="axes fraction",
            ha="right",
            va="top",
            ma="right",
            fontsize="x-small",
        )

    plot_norms(ax2, labels, ar, rng=rng, set_ylabel=norm_ylabel)


def main():

    root = Path("../../experiments/")
    dataset = sys.argv[2]
    stylef = "../project.mplstyle"

    crop_scales = [0.08, 0.2, 0.4, 0.8, 1]
    paths = []
    for cs in crop_scales:
        if cs == 0.2:
            ds_ = dataset
        else:
            ds_ = f"{dataset}:crop_scale_lo={cs}"

        prefix = root / ds_ / "dl"
        path = (
            prefix / names.default_train(metric="euclidean") / names.finetune()
        )
        paths.append(path)

    redo.redo_ifchange_slurm(
        [p / "default.run" for p in paths],
        name=[f"crop {cs}" for cs in crop_scales],
        partition="gpu-2080ti",
        time_str="22:30:00",
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
        fig, axxs = plt.subplots(
            nrows=2, ncols=len(crop_scales), figsize=(5.5, 3), squeeze=False
        )
        ylab = True
        for path, cs, axs in zip(paths, crop_scales, axxs.T):
            title = f"$c = {cs}$"
            plot_one(path, axs, title, norm_ylabel=ylab)
            ylab = False

        add_letters(axxs)

    metadata = get_default_metadata()
    metadata[
        "Title"
    ] = f"Different crop parameters and visualization for {dataset}"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
