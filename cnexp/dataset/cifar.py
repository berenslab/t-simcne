from .base import DatasetBase
from ..imagedistortions import get_transforms

import torch
from torch.utils.data import Dataset
from torchvision import datasets as tvdatasets


def load_cifar10(download=True, **kwargs):
    # hardcode dataset mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = get_transforms(mean, std, size=(32, 32), setting="contrastive")

    return tvdatasets.CIFAR10(transform=transform, download=download, **kwargs)


def load_cifar100(**kwargs):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    transform = get_transforms(mean, std, size=(32, 32), setting="contrastive")

    # need to make a dataset that returns two transforms of an image

    return tvdatasets.CIFAR100(transform=transform, **kwargs)


class CIFAR10(DatasetBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs

    def compute(self):
        self.cifar = load_cifar10(root=self.outdir / "cifar10", **self.kwargs)

    def save(self):
        torch.save(self.cifar, self.outdir / "dataset.pt")


class CIFAR100(Dataset):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs

    def compute(self):
        self.cifar = load_cifar100(root=self.outdir / "cifar100", **self.kwargs)
