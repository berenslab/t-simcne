#!/usr/bin/env python
# -*- mode: python-mode -*-

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cnexp import dispatch, redo

if __name__ == "__main__":

    c10 = Path("../experiments/cifar/")
    c100 = Path("../experiments/cifar100/")

    redo.redo_ifchange([c / "dataset.pt" for c in [c10, c100]])
    datasets = [torch.load(c / "dataset.pt") for c in [c10, c100]]

    n_samples = 5
    fig, axs = plt.subplots(2, 2 * n_samples, figsize=(2 * n_samples, 2))
    for i, (dataset, _) in enumerate(itertools.product(datasets, range(n_samples))):
        x1, x2 = dataset[i]
        axs[0, i].imshow(x1.mean(0), cmap="gray")
        axs[1, i].imshow(x2.mean(0), cmap="gray")

    [ax.set_axis_off() for ax in axs.flat]

    fig.savefig(sys.argv[3], format="pdf")
