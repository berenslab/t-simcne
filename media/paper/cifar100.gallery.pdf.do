#!/usr/bin/env python

import inspect
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cnexp import names, redo
from cnexp.plot import add_letters, get_default_metadata
from cnexp.plot.scalebar import add_scalebar_frac


def load(dir, extract_name="embeddings/post.npy"):
    with zipfile.ZipFile(dir / "intermediates.zip") as zipf:
        with zipf.open(extract_name) as f:
            return np.load(f)


superclasses = {
    "aquatic mammals": [4, 30, 55, 72, 95],
    "fish": [1, 32, 67, 73, 91],
    "flowers": [54, 62, 70, 82, 92],
    "food containers": [9, 10, 16, 28, 61],
    "fruit and vegetables": [0, 51, 53, 57, 83],
    "household electrical devices": [22, 39, 40, 86, 87],
    "household furniture": [5, 20, 25, 84, 94],
    "insects": [6, 7, 14, 18, 24],
    "large carnivores": [3, 42, 43, 88, 97],
    "large man-made outdoor things": [12, 17, 37, 68, 76],
    "large natural outdoor scenes": [23, 33, 49, 60, 71],
    "large omnivores and herbivores": [15, 19, 21, 31, 38],
    "medium-sized mammals": [34, 63, 64, 66, 75],
    "non-insect invertebrates": [26, 45, 77, 79, 99],
    "people": [2, 11, 35, 46, 98],
    "reptiles": [27, 29, 44, 78, 93],
    "small mammals": [36, 50, 65, 74, 80],
    "trees": [47, 52, 56, 59, 96],
    "vehicles 1": [8, 13, 48, 58, 90],
    "vehicles 2": [41, 69, 81, 85, 89],
}

label_map = [float("inf")] * 100
[
    label_map.__setitem__(v, i)
    for i, vals in enumerate(superclasses.values())
    for v in vals
]
assert float("inf") not in label_map

superclass_names = list(superclasses.keys())


def fine_to_coarse(label_idx: int) -> int:
    """map the cifar100 fine label to its superclass label"""
    return label_map[label_idx]


def add_cifar100_legend(ax, cmap="tab20"):
    cm = plt.get_cmap(cmap)

    names = []
    span = 2
    for name in superclass_names:
        words = name.split(" ")
        wordpairs = [
            " ".join(words[i : i + span]) for i in range(0, len(words), span)
        ]
        name_fmt = "\n".join(wordpairs)

        # break up long words
        if name_fmt == "medium-sized mammals":
            name_fmt = "medium-sized\nmammals"
        elif name_fmt == "non-insect invertebrates":
            name_fmt = "non-insect\ninvertebrates"
        elif name_fmt == "household furniture":
            name_fmt = "household\nfurniture"
        elif name_fmt == "household electrical\ndevices":
            name_fmt = "household elec-\ntrical devices"
        names.append(name_fmt)

    markers = [
        mpl.lines.Line2D(
            [],
            [],
            label=name,
            color=cm(i),
            ls="",
            marker=mpl.rcParams["scatter.marker"],
            # markersize=mpl.rcParams["font.size"],
            markersize=5,
        )
        for i, name in enumerate(names)
    ]
    legend = ax.legend(
        handles=markers,
        ncol=2,
        fontsize="small",
        loc="center",
        handletextpad=0.1,
        columnspacing=0,
        borderaxespad=0,
    )
    legend.get_frame().set_linewidth(0.4)
    return legend


def main():

    root = Path("../../experiments/")
    prefix = root / "cifar100" / "dl"
    stylef = "../project.mplstyle"

    pdict = dict(
        euc=prefix / names.default_train(out_dim=2),
        # euc_5k=prefix / names.default_train(out_dim=2, n_epochs=5000),
        # needs to be plotted differently, with Hammer projection
        # cos=prefix / names.default_train(out_dim=3, metric="cosine"),
        ft_cos=prefix
        / names.default_train(metric="cosine")
        / names.finetune(),
        ft_euc=prefix / names.default_train() / names.finetune(),
    )
    titles = dict(
        euc="Euclidean",
        euc_5k="Euclidean (5000 epochs)",
        cos="Cosine",
        ft_euc=r"Euclidean $\to$ Euclidean",
        ft_cos=r"Cosine $\to$ Euclidean",
        # euc="Trained from scratch\n",
        # euc_5k="Euclidean 5000 epochs",
        # cos="Cosine",
        # ft_euc="Fine-tuned\n",
        # ft_cos="Fine-tuned\nfrom default SimCLR",
    )
    redo.redo_ifchange(prefix / "dataset.pt")
    # redo.redo_ifchange_slurm(
    #     [d / "intermediates.zip" for d in pdict.values()],
    #     name=[f"{key}-c100" for key in pdict.keys()],
    #     partition="gpu-2080ti",
    #     time_str="24:00:00",
    # )
    redo.redo_ifchange(
        [
            stylef,
            inspect.getfile(add_scalebar_frac),
            inspect.getfile(add_letters),
            inspect.getfile(names),
        ]
    )

    labels_fine = load(pdict["ft_euc"], "labels.npy")
    labels = [fine_to_coarse(lbl) for lbl in labels_fine]

    seed = 44
    with plt.style.context(stylef):
        fig, axs = plt.subplots(
            nrows=1,
            ncols=len(pdict) + 1,
            figsize=(5.5, 1.5),
            constrained_layout=True,
        )
        for key, ax in zip(pdict.keys(), axs):
            ar = load(pdict[key])
            ax.scatter(
                ar[:, 0],
                ar[:, 1],
                c=labels,
                alpha=0.5,
                cmap="tab20",
                rasterized=True,
            )
            add_scalebar_frac(ax)
            ax.set_title(titles[key])

            from cnexp.eval.knn import knn_acc
            from sklearn.model_selection import train_test_split

            split = train_test_split(
                ar, labels, test_size=10000, stratify=labels, random_state=seed
            )
            knn1 = knn_acc(*split, metric="euclidean")
            split = train_test_split(
                ar,
                labels_fine,
                test_size=10000,
                stratify=labels_fine,
                random_state=seed,
            )
            knn2 = knn_acc(*split, metric="euclidean")
            print(f"{key:15s}: {knn1:.0%}, {knn2:.0%}", file=sys.stderr)

        add_letters(axs[:-1])

        ax = axs[-1]
        ax.set_axis_off()
        add_cifar100_legend(ax)

    metadata = get_default_metadata()
    metadata["Title"] = f"Various visualizations of the {sys.argv[2]} dataset"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
