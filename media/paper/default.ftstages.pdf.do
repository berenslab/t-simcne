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


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    # train_kwargs = dict(device=1)
    train_kwargs = dict()
    default = prefix / names.default_train(
        metric="euclidean",
        train_kwargs=train_kwargs,
    )
    last = default / names.finetune(train_kwargs=train_kwargs)
    ft_lastlin = last.parent
    while not ft_lastlin.name.startswith("train"):
        ft_lastlin = ft_lastlin.parent

    path_dict = dict(ft_lin=ft_lastlin, last=last)
    fnames = [path / "intermediates.zip" for path in path_dict.values()]
    redo.redo_ifchange_slurm(
        last / "default.run",
        name="ftstages",
        partition="gpu-v100-preemptable",
        time_str="18:30:00",
    )
    redo.redo_ifchange(
        fnames
        + [
            stylef,
            inspect.getfile(add_scalebar_frac),
            inspect.getfile(add_letters),
            inspect.getfile(names),
        ]
    )

    keys = ["pre", "ft_lin", "final"]
    embs = dict()
    with zipfile.ZipFile(path_dict["ft_lin"] / "intermediates.zip") as zipf:
        with zipf.open("embeddings/pre.npy") as f:
            embs["pre"] = np.load(f)

        with zipf.open("labels.npy") as f:
            labels = np.load(f)
    for key, p in zip(keys[1:], path_dict.values()):
        with zipfile.ZipFile(p / "intermediates.zip") as zipf:
            with zipf.open("embeddings/post.npy") as f:
                embs[key] = np.load(f)

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(2.5, 0.75),
            constrained_layout=True,
        )

        for ax, (key, ar) in zip(axs, embs.items()):
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            ax.set_title(key)
            add_scalebar_frac(ax)
        add_letters(axs)

    metadata = get_default_metadata()
    metadata["Title"] = "Different stages of finetuning for SimCLR to 2D"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
