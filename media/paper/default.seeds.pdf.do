#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, plot, redo


def load(dir, extract_name="embeddings/post.npy"):
    with zipfile.ZipFile(dir / "intermediates.zip") as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


def main():

    root = Path("../../experiments/")
    prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    rng = np.random.default_rng(511622144)
    seeds = [None, *rng.integers(10_000, size=2)]

    paths = [
        prefix
        / names.default_train(random_state=seed)
        / names.finetune(random_state=seed)
        for seed in seeds
    ]

    redo.redo_ifchange_slurm(
        [path / "intermediates.zip" for path in paths],
        name=[f"seed-{seed}" for seed in seeds],
        partition="gpu-2080ti-preemptable",
        time_str="18:30:00",
    )
    redo.redo_ifchange(
        [
            stylef,
            inspect.getfile(plot.add_scalebar_frac),
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    anchor = load(paths[-1])
    labels = load(paths[0], "labels.npy")

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            ncols=len(paths),
            figsize=(5.5, 1.85),
            constrained_layout=True,
        )
        for seed, path, ax in zip(seeds, paths, axs):
            ar = load(path)
            ar = plot.flip_maybe(ar, anchor=anchor)
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            plot.add_scalebar_frac(ax)

        plot.add_letters(axs)

    metadata = plot.get_default_metadata()
    metadata["Title"] = (
        f"The same visualization of the {sys.argv[2]} dataset "
        f" with {len(seeds)} different seeds"
    )
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
