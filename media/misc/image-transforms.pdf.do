#!/usr/bin/env python

import inspect
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cnexp import names, plot, redo
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plot_imgs(axs, dataset):
    rng_ = np.random.default_rng(4534234)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    invnorm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / s for s in std]
            ),
            transforms.Normalize(mean=[-m for m in mean], std=[1.0, 1.0, 1.0]),
        ]
    )

    axxs = axs.reshape(-1, 3)
    for ltr, axs in zip(string.ascii_lowercase, axxs):
        seed = rng_.integers(2**64, dtype="uint")
        rng = np.random.default_rng(seed)
        torch.manual_seed(rng.integers(2**64, dtype="uint"))
        ix = rng.integers(len(dataset))

        orig_im, label = dataset.dataset[ix]
        (im1, im2), label = dataset[ix]
        im1 = to_pil_image(invnorm(im1))
        im2 = to_pil_image(invnorm(im2))

        for i, (ax, im) in enumerate(zip(axs, (orig_im, im1, im2))):
            ax.imshow(im)
            ax.set_axis_off()
            plot.add_lettering(ax, f"{ltr}.{i}")
        eprint(ltr, seed)


def main():

    root = Path("../../experiments/cifar")
    stylef = "../project.mplstyle"

    redo.redo_ifchange(
        [
            root / "dataset.pt",
            stylef,
            inspect.getfile(plot),
            inspect.getfile(names),
        ]
    )

    dataset = torch.load(root / "dataset.pt")["train_contrastive"]

    with plt.style.context(stylef):
        fig, axs = plt.subplots(10, 6, figsize=(5.5, 11))

        plot_imgs(axs, dataset)

    metadata = plot.get_default_metadata()
    metadata["Title"] = "Sample images with transformations applied"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
