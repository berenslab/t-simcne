#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, redo
from cnexp.plot import add_letters
from cnexp.plot.scalebar import add_scalebar_frac

if __name__ == "__main__":

    prefix = Path("../../experiments/cifar/dl/")
    stylef = "../project.mplstyle"

    path_dict = dict(
        normal=prefix / "model:out_dim=2/sgd/lrcos/infonce/train/",
        epoch5k=prefix
        / "model:out_dim=2/sgd/lrcos:n_epochs=5000/infonce/train/",
        finetune=prefix / names.default_train() / names.finetune(),
    )
    fnames = [path / "intermediates.zip" for path in path_dict.values()]
    redo.redo_ifchange(fnames + [stylef, inspect.getfile(add_scalebar_frac)])

    for key, fname in zip(path_dict.keys(), fnames):
        with zipfile.ZipFile(fname) as zipf:
            with zipf.open("embeddings/post.npy") as f:
                ar = np.load(f)
                path_dict[key] = ar

            with zipf.open("labels.npy") as f:
                labels = np.load(f)

    blocksize = 1.25
    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            1,
            len(path_dict),
            figsize=(len(path_dict) * blocksize, blocksize + 0.05),
            constrained_layout=True,
        )

        for ax, (key, ar) in zip(axs.flat, path_dict.items()):
            ax.scatter(
                ar[:, 0], ar[:, 1], c=labels, alpha=0.5, rasterized=True
            )
            ax.set_title(key)
            add_scalebar_frac(ax)

        add_letters(axs)
    fig.savefig(sys.argv[3], format="pdf")
