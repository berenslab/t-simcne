#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openTSNE

# import torch
from cnexp import plot, redo

# from cnexp.callback import to_features


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plot_supervised(fig, ax, ar, labels, title, rng):
    seed = rng.integers(0, 2**32)
    tsne = openTSNE.TSNE(n_jobs=-1, random_state=seed)
    Y = tsne.fit(ar)
    ax.scatter(Y[:, 0], Y[:, 1], c=labels, alpha=0.5, rasterized=True)
    ax.set_title(title)


def main():

    root = Path("../../")
    stylef = "../project.mplstyle"
    paths = (
        root / "experiments/cifar/dl/model:out_dim=10/"
        "sgd:lr=0.1/lrlin:n_epochs=100/ce_loss/suptrain2",
        root / "experiments/cifar100/dl/model:out_dim=100/"
        "sgd:lr=0.1/lrlin:n_epochs=100/ce_loss/suptrain2",
    )
    titles = ["CIFAR-10", "CIFAR-100"]

    redo.redo_ifchange_slurm(
        [p / "intermediates.zip" for p in paths],
        name=["sup-c10", "sup-c100"],
        time_str="18:00:00",
        partition="gpu-2080ti",
    )
    accfiles = [p / "score.txt" for p in paths]
    redo.redo_ifchange(
        accfiles
        + [
            # zfname,
            stylef,
            inspect.getfile(plot),
        ]
    )

    rng = np.random.default_rng(520221)

    with plt.style.context(stylef):
        fig, axs = plt.subplots(1, 2, figsize=(3.66, 1.75))

        for path, title, accf, ax in zip(paths, titles, accfiles, axs):
            eprint("Start", title)
            # model = torch.load(path / "model.pt", map_location="cpu")["model"]
            # dataset = torch.load(path.parent / "dataset.pt")[
            #     "train_plain_loader"
            # ]
            with zipfile.ZipFile(path / "out/intermediates.zip") as zf:
                with zf.open("backbone_embeddings/post.npy") as f:
                    ar = np.load(f)

                with zf.open("labels.npy") as f:
                    labels = np.load(f)
            # feat, bb_feat, labels = to_features(model, dataset, "cpu")
            plot_supervised(fig, ax, ar, labels, title, rng)
            plot.add_scalebar_frac(ax)

            # acc = float(accf.read_text())
            # acctxt = f"acc = {acc:.1%}"
            # ax.text(
            #     1,
            #     1,
            #     acctxt,
            #     fontsize="medium",
            #     ha="right",
            #     va="top",
            #     transform=ax.transAxes,
            # )

        plot.add_letters(axs)
    metadata = plot.get_default_metadata()
    metadata["Title"] = "t-SNE of supervised resnet"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
