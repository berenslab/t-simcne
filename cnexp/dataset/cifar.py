from .base import DatasetBase
from ..imagedistortions import TransformedPairDataset, get_transforms

import contextlib
import inspect

import torch
from torchvision import datasets as tvdatasets


def load_cifar10(download=True, **kwargs):
    # hardcode dataset mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = get_transforms(mean, std, size=(32, 32), setting="contrastive")

    dataset = tvdatasets.CIFAR10(
        download=download,
        **kwargs,
    )

    # need to make a dataset that returns two transforms of an image
    return TransformedPairDataset(dataset, transform)


def load_cifar100(download=True, **kwargs):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform = get_transforms(mean, std, size=(32, 32), setting="contrastive")

    dataset = tvdatasets.CIFAR100(
        download=download,
        **kwargs,
    )

    # need to make a dataset that returns two transforms of an image
    return TransformedPairDataset(dataset, transform)


class CIFAR10(DatasetBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs

    def get_deps(self):
        supdeps = super().get_deps()
        return supdeps + [inspect.getfile(TransformedPairDataset)]

    def compute(self):
        with open(self.outdir / "stdout.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                self.cifar = load_cifar10(root=self.outdir / "cifar10", **self.kwargs)

    def save(self):
        self.save_lambda(
            self.outdir / "dataset.pt",
            self.cifar,
            lambda file, data: torch.save(data, file),
        )


class CIFAR100(CIFAR10):
    def compute(self):
        with open(self.outdir / "stdout.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                self.cifar = load_cifar100(root=self.outdir / "cifar100", **self.kwargs)
