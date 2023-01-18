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


def plot_loss(ax, paths):
    title = ""
    if any(p.startswith("train") for p in paths[0].parent.parts):
        pretrain = paths[0].parent
        while not pretrain.name.startswith("train"):
            pretrain = pretrain.parent

        others = []
        par = pretrain.parent
        while any(p.startswith("train") for p in par.parts):
            o = par.parent
            while not o.name.startswith("train"):
                o = o.parent
            others.append(o)
            par = o.parent
        paths = list(reversed(others)) + [pretrain] + paths
        knnfiles = [p / "knn/score.txt" for p in paths]
        lossfiles = [p / "losses.csv" for p in paths]
        redo.redo_ifchange(
            knnfiles + lossfiles,
        )
        score = float((pretrain / "knn/score.txt").read_text())
        title += f"first stage knn = {score:.1%}"
    else:
        knnfiles = [p / "knn/score.txt" for p in paths]
        lossfiles = [p / "losses.csv" for p in paths]

    dfs = [pd.read_csv(f)["mean"] for f in lossfiles]
    loss = pd.concat(dfs, ignore_index=True)
    ax.plot(loss, c="xkcd:dark grey")

    n_epochs = np.cumsum([len(df) for df in dfs])
    scores = (float(Path(f).read_text()) for f in knnfiles)
    yheights = np.linspace(0.4, 0.05, num=len(n_epochs))
    arrowd = dict(
        arrowstyle="->", color="xkcd:slate grey", relpos=(1, 0.5), shrinkA=0.5
    )
    for epoch, score, y in zip(n_epochs, scores, yheights):
        if epoch != n_epochs[-1]:
            ax.axvline(epoch, c="xkcd:light grey", zorder=1)
        scoretxt = f"{score:.1%}"
        ax.annotate(
            scoretxt,
            xy=(epoch, loss[epoch - 1]),
            xycoords="data",
            xytext=(0.1, y),
            textcoords="axes fraction",
            arrowprops=arrowd,
            va="bottom",
            ha="left",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    # ax.set_title(title, loc="right", fontsize="x-small")
    ax.set_title("knn acc. per stage")

    # ticks = ax.get_yticks()
    # ax.spines.left.set_bounds(ticks[0], ticks[-1])
    ax.spines.bottom.set_bounds(loss.index[0], loss.index[-1])
    ax.set_xlim(right=loss.index[-1])

    return paths


def plot_finetune_stages(ft_path, axs, titles, knn=True, loss=True):
    """Plot the three stages within ft_path

    Plots the different stages of training for finetuning.  (1) just
    after changing the last layer to 2D, before starting the
    optimization.  (2) after optimizing the last linear layer only.
    (3) after optimizing the entire network.

    """
    assert len(titles) == 3, f"Expected three titles, but got {titles = }"

    rng = np.random.default_rng(55**2 - 1)

    final = ft_path
    ft_lin = ft_path.parent
    while not ft_lin.name.startswith("train"):
        ft_lin = ft_lin.parent

    initial = ft_lin.parent
    while not initial.name.startswith("train"):
        initial = initial.parent

    paths = [initial, ft_lin, final]
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

    with zipfile.ZipFile(ft_lin / "intermediates.zip") as zipf:
        with zipf.open("embeddings/pre.npy") as f:
            pre_emb = np.load(f)

        with zipf.open("labels.npy") as f:
            labels = np.load(f)

    with zipfile.ZipFile(ft_lin / "intermediates.zip") as zipf:
        with zipf.open("embeddings/post.npy") as f:
            ft_emb = np.load(f)
    with zipfile.ZipFile(final / "intermediates.zip") as zipf:
        with zipf.open("embeddings/post.npy") as f:
            final_emb = np.load(f)

    embs = [pre_emb, ft_emb, final_emb]
    set_ylabel = True
    for (ax1, ax2), ar, title in zip(axs.T, embs, titles):
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

        plot_norms(ax2, labels, ar, rng=rng, set_ylabel=set_ylabel)
        set_ylabel = False

    all_paths = plot_loss(axs[0, -1], paths)
    if "circle_trained" in ft_path.parts:
        n_stage = 0
        stage_str = 2
    else:
        n_stage = 2
        stage_str = n_stage
    p = all_paths[n_stage]
    # p = Path("../../experiments/fixed/circle_trained")
    redo.redo_ifchange(p / "intermediates.zip")
    with zipfile.ZipFile(p / "intermediates.zip") as zf:
        with zf.open("embeddings/post.npy") as f:
            a = np.load(f)
    plot_norms(
        axs[-1, -1],
        labels,
        a,
        rng=rng,
        set_ylabel=False,
        title=f"stage {stage_str} ({a.shape[1]}D)",
    )


def main():

    root = Path("../../experiments/")
    dataset = sys.argv[2]
    if dataset == "tiny":
        prefix = root / "tiny" / "dl:num_workers=48"
    else:
        prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    if dataset == "cifar":
        # use the fixed one so it does not get rerun
        circle_train = root / "fixed/circle_trained"
    else:
        circle_train = (
            prefix
            / names.default_train(metric="cosine")
            # cosine training with a circle regularizer
            / "sgd:lr=0.0012/lrcos:n_epochs=100"
            / "infonce:metric=cosine:reg_coef=1:reg_radius=1/train"
        )

    ft = (
        circle_train
        # Euclidean training in high dim
        / "sgd:lr=0.0012/lrcos:n_epochs=400/infonce:temperature=200/train"
        / names.finetune(ft_loss="infonce:temperature=200", ft_lr=1.2e-5)
    )

    redo.redo_ifchange_slurm(
        ft / "default.run",
        name="ftstages",
        partition="gpu-v100-preemptable",
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

    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=2, ncols=4, figsize=(5.5, 3), squeeze=False
        )
        titles = [
            "Before optimization",
            "Linear fine-tuning",
            "Final fine-tuning",
        ]
        plot_finetune_stages(ft, axs, titles)

        add_letters(axs)

    metadata = get_default_metadata()
    metadata["Title"] = "Different stages of finetuning for SimCLR to 2D"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
