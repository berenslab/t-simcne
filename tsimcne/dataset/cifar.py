import contextlib
import inspect

from torchvision import datasets as tvdatasets

from ..imagedistortions import TransformedPairDataset, get_transforms
from .base import DatasetBase
from .load_torchvision_dataset import load_torchvision_dataset


def load_cifar10(**kwargs):
    # hardcode dataset mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    size = (32, 32)

    return load_torchvision_dataset(
        tvdatasets.CIFAR10, mean, std, size, **kwargs
    )


def load_cifar100(download=True, train=None, **kwargs):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    size = (32, 32)

    return load_torchvision_dataset(
        tvdatasets.CIFAR100, mean, std, size, **kwargs
    )


class CIFAR10(DatasetBase):
    def get_deps(self):
        supdeps = super().get_deps()
        deps = [
            inspect.getfile(get_transforms),
            inspect.getfile(load_torchvision_dataset),
            inspect.getfile(TransformedPairDataset),
        ]
        return supdeps + deps

    def compute(self):
        with open(self.outdir / "stdout.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    self.data_sd = load_cifar10(
                        root=self.outdir / "cifar10", **self.kwargs
                    )


class CIFAR100(CIFAR10):
    def compute(self):
        with open(self.outdir / "stdout.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    self.data_sd = load_cifar100(
                        root=self.outdir / "cifar100", **self.kwargs
                    )
