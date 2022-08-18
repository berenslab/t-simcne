#!/usr/bin/env python
# -*- mode: python-mode -*-

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cnexp import redo

if __name__ == "__main__":

    c10 = Path("../../experiments/cifar/")
    c100 = Path("../../experiments/cifar100/")

    redo.redo_ifchange([c / "dataset.pt" for c in [c10, c100]])
    datasets = [
        torch.load(c / "dataset.pt")["train_contrastive"] for c in [c10, c100]
    ]

    n_samples = 5
    fig, axs = plt.subplots(
        2, 2 * n_samples, figsize=(2 * n_samples, 2.5), constrained_layout=True
    )
    for i, (dataset, _) in enumerate(
        itertools.product(datasets, range(n_samples))
    ):
        xs, (orig, label) = dataset[i]
        xs = [x.transpose(0, 2).transpose(1, 0) for x in xs]
        xs = [x - x.min() for x in xs]
        xs = [x / x.max() for x in xs]
        assert xs[0].shape == (32, 32, 3)

        axs[0, i].set_title(dataset.dataset.classes[label])
        axs[0, i].imshow(xs[0])
        axs[1, i].imshow(xs[1])

    [ax.set_axis_off() for ax in axs.flat]

    fig.savefig(sys.argv[3], format="pdf")
