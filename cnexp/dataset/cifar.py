from .base import DatasetBase
from torch.utils.data import Dataset
from torchvision import datasets as tvdatasets


def load_cifar10(download=True, **kwargs):
    # hardcode dataset mean and std
    mean = 0
    std = 1
    # import imagedistortions and use that one here.
    transform = None  #

    return tvdatasets.CIFAR10(transform=transform, download=download, **kwargs)


def load_cifar100(**kwargs):
    # import imagedistortions and use that one here.
    transform = None  #

    return tvdatasets.CIFAR100(transform=transform, **kwargs)


class CIFAR10(DatasetBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs

    def compute(self):
        self.cifar = load_cifar10(root=self.outdir / "cifar10", **self.kwargs)


class CIFAR100(Dataset):
    pass
