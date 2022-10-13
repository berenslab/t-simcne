#!/usr/bin/env python

import inspect
import os
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, plot, redo


def load(dir, extract_name="embeddings/post.npy"):
    zipname = dir / "intermediates.zip"
    if not zipname.exists():
        redo.redo_ifchange(zipname)

    with zipfile.ZipFile(zipname) as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    budget_schedules = [[400, 25, 75], [775, 25, 200], [1000, 50, 450]]
    rng = np.random.default_rng(511622144)
    seeds = [None, *rng.integers(10_000, size=2)]

    # remove last one, is done elsewhere.
    # budget_schedules = budget_schedules[:-1]
    pdict = {
        sum(b): prefix
        / names.default_train(n_epochs=b[0], random_state=seeds[-1])
        / names.finetune(
            llin_epochs=b[1], ft_epochs=b[2], random_state=seeds[-1]
        )
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
    # redo.redo_ifchange(
    #     # let's compute them on the headnode, it doesn't take long
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
        for key, ax, budget in zip(pdict.keys(), axs.flat, budget_schedules):
            ar = load(pdict[key])
            ar = plot.flip_maybe(ar, anchor=anchor)
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            plot.add_scalebar_frac(ax)
            default = "" if key != 1500 else " (default)"
            ax.set_title(
                f"{key} epochs{default}\n({', '.join(str(b) for b in budget)})"
            )

        plot.add_letters(axs)

    metadata = plot.get_default_metadata()
    metadata["Title"] = f"Various visualizations of the {sys.argv[2]} dataset"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
