#!/usr/bin/env python

import inspect
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnexp import names, redo
from cnexp.plot import add_letters, get_default_metadata
from cnexp.plot.scalebar import add_scalebar_frac


def load(dir, extract_name="embeddings/post.npy"):
    zipname = dir / "intermediates.zip"
    if not zipname.exists():
        redo.redo_ifchange(zipname)

    with zipfile.ZipFile(zipname) as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


def flip_maybe(
    anchor,
    other,
    n_samples=10_000,
    return_orig=True,
):

    flipx = np.cov(anchor[:, 0], other[:, 0])[0, 1]
    flipy = np.cov(anchor[:, 1], other[:, 1])[0, 1]

    flip = np.sign([flipx, flipy], dtype=other.dtype)

    if return_orig:
        return anchor, other * flip
    else:
        return other * flip


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    budget_schedules = [[400, 25, 75], [775, 25, 200], [1000, 50, 450]]
    # remove last one, is done elsewhere.
    # budget_schedules = budget_schedules[:-1]
    pdict = {
        sum(b): prefix
        / names.default_train(n_epochs=b[0])
        / names.finetune(llin_epochs=b[1], ft_epochs=b[2])
        for b in budget_schedules
    }
    common_prefix = Path(os.path.commonpath(pdict.values()))
    # redo.redo_ifchange([common_prefix / f for f in ["model.pt", "dataset.pt"]])
    # redo.redo_ifchange_slurm(
    #     [d / "intermediates.zip" for d in pdict.values()],
    #     name=[f"budget-{sum(b)}" for b in budget_schedules],
    #     partition="gpu-2080ti",
    #     time_str="18:30:00",
    # )
    knn_scores = [
        d / "knn:metric=euclidean:layer=Z/score.txt" for d in pdict.values()
    ]
    # redo.redo_ifchange(
    #     # let's compute them on the headnode, it doesn't take long
    #     knn_scores
    #     + [d / "losses.csv" for d in pdict.values()]
    #     + [
    #         stylef,
    #         inspect.getfile(add_scalebar_frac),
    #         inspect.getfile(add_letters),
    #         inspect.getfile(names),
    #     ]
    # )

    labels = load(pdict[1500], "labels.npy")
    anchor = load(pdict[1500])

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(5.5, 1.75),
            constrained_layout=True,
        )
        for key, ax, budget, knnf in zip(
            pdict.keys(), axs.flat, budget_schedules, knn_scores
        ):
            ar = load(pdict[key])
            anchor, ar = flip_maybe(anchor, ar)
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            add_scalebar_frac(ax)
            default = "" if key != 1500 else " (default)"
            ax.set_title(
                f"{key} epochs{default}\n({', '.join(str(b) for b in budget)})"
            )

            score = float(knnf.read_text())
            knn = f"$k$nn = {score:.1%}"
            loss_df = pd.read_csv(pdict[key] / "losses.csv")["mean"]
            loss = f"final loss {loss_df.tail(1).item():.1f}"
            ax.set_title(f"{knn}\n{loss}", loc="right", fontsize="small")


        add_letters(axs)

    metadata = get_default_metadata()
    metadata["Title"] = f"Various visualizations of the {sys.argv[2]} dataset"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
