#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cnexp import plot, redo

if __name__ == "__main__":

    prefix = Path("../../experiments/cifar/dl/")
    stylef = "../project.mplstyle"

    fname = (
        prefix / "model:out_dim=3/sgd/lrcos/infonce:metric=cosine"
        "/train/intermediates.zip"
    )
    # fnames = [path / "intermediates.zip" for path in path_dict.values()]
    # redo.redo_ifchange_slurm(
    #     fnames, partition="gpu-2080ti-preemptable", time_str="17:00:00"
    # )
    redo.redo_ifchange([stylef, inspect.getfile(plot)])

    with zipfile.ZipFile(fname) as zipf:
        with zipf.open("embeddings/post.npy") as f:
            ar = np.load(f).astype(float)

        with zipf.open("labels.npy") as f:
            labels = np.load(f)

    norm = (ar**2).sum(1) ** 0.5
    ar /= norm[:, None]

    x, y, z = ar.T
    lat = np.arccos(x) - np.pi / 2
    lat *= 2
    lon = np.arctan(y / z)

    blocksize = 5.5
    with plt.style.context(stylef):
        fig = plt.figure(
            figsize=(4, 1.85),
            constrained_layout=False,
        )
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(
            ar[:, 0],
            ar[:, 1],
            ar[:, 2],
            c=labels,
            alpha=0.5,
            s=1,
            marker=".",
            rasterized=True,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            width=0,
            labelbottom=False,
            labeltop=False,
            labelright=False,
            labelleft=False,
            grid_linewidth=plt.rcParams["axes.linewidth"],
        )
        ax.w_xaxis.gridlines.set_lw(plt.rcParams["axes.linewidth"])
        ax.w_yaxis.gridlines.set_lw(plt.rcParams["axes.linewidth"])
        ax.w_zaxis.gridlines.set_lw(plt.rcParams["axes.linewidth"])
        ax.set_box_aspect((1, 1, 1))

        ax = fig.add_subplot(1, 2, 2, projection="hammer")
        ax.scatter(lat, lon, c=labels, alpha=0.5, rasterized=True)
        ax.set_axis_off()

        # plot.add_letters(fig.get_axes())
        kwargs = plot.get_lettering_fprops()
        kwargs.update(ha="left", va="top")
        fig.text(0.05, 0.9, "a", **kwargs)
        fig.text(0.55, 0.9, "b", **kwargs)
        fig.subplots_adjust(0, 0, 1, 1, wspace=0.05)

    metadata = plot.get_default_metadata()
    metadata["Title"] = "Map projection of 3D SimCLR embedding"
    fig.savefig(sys.argv[3], format="pdf")
