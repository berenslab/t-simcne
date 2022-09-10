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


def load(dir, extract_name="embeddings/post.npy"):
    with zipfile.ZipFile(dir / "intermediates.zip") as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    budget_schedules = [[400, 25, 75], [775, 25, 200], [1000, 50, 450]]
    pdict = {
        sum(b): prefix
        / names.default_train(n_epochs=b[0])
        / names.finetune(llin_epochs=b[1], ft_epochs=b[2])
        for b in budget_schedules
    }
    redo.redo_ifchange_slurm(
        [d / "intermediates.zip" for d in pdict.values()],
        name=[f"budget-{sum(b)}" for b in budget_schedules],
        partition="gpu-2080ti-preemptable",
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

    labels = load(pdict["ft_euc"], "labels.npy")

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=len(pdict),
            figsize=(5.5, 2),
            constrained_layout=True,
        )
        for key, ax, budget in zip(pdict.keys(), axs, budget_schedules):
            ar = load(pdict[key])
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            add_scalebar_frac(ax)
            lbl = key if key != 1000 else "default"
            ax.set_title(f"Budget {lbl}\n({', '.join(budget)})")

        add_letters(axs)

    metadata = get_default_metadata()
    metadata["Title"] = f"Various visualizations of the {sys.argv[2]} dataset"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
