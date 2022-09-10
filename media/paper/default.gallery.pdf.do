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

    pdict = dict(
        euc=prefix / names.default_train(out_dim=2),
        euc_5k=prefix / names.default_train(out_dim=2, n_epochs=5000),
        # needs to be plotted differently, with Hammer projection
        # cos=prefix / names.default_train(out_dim=3, metric="cosine"),
        ft_euc=prefix / names.default_train() / names.finetune(),
        ft_cos=prefix
        / names.default_train(metric="cosine")
        / names.finetune(),
    )
    titles = dict(
        euc="Euclidean",
        euc_5k="Euclidean 5000 epochs",
        cos="Cosine",
        ft_euc="Fine-tuned",
        ft_cos="Fine-tuned cosine",
    )
    redo.redo_ifchange_slurm(
        [d / "intermediates.zip" for d in pdict.values()],
        name="gallery",
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
            figsize=(5.5, 1.5),
            constrained_layout=True,
        )
        for key, ax in zip(pdict.keys(), axs):
            ar = load(pdict[key])
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            add_scalebar_frac(ax)
            ax.set_title(titles[key])

        add_letters(axs)

    metadata = get_default_metadata()
    metadata["Title"] = f"Various visualizations of the {sys.argv[2]} dataset"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
