import contextlib
import inspect

import medmnist
from torch.utils.data import ConcatDataset
from torchvision import transforms

from ..imagedistortions import TransformedPairDataset
from .base import DatasetBase


def get_transforms(mean, std, size, setting):
    normalize = transforms.Normalize(mean=mean, std=std)

    if setting == "contrastive":
        return transforms.Compose(
            [
                transforms.RandomRotation(180),
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "train_linear_classifier":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "test_linear_classifier":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown transformation setting {setting!r}")


def load_dermamnist(**kwargs):
    # hardcode dataset mean and std
    mean = (0.763685025651589, 0.53826926027722, 0.5614717289800738)
    std = (0.13714251162834182, 0.15401810261738114, 0.16897525910549463)
    size = (28, 28)

    return load_medmnist_dataset(
        medmnist.DermaMNIST, mean, std, size, **kwargs
    )


class DermaMNIST(DatasetBase):
    def get_deps(self):
        supdeps = super().get_deps()
        deps = [
            inspect.getfile(get_transforms),
            inspect.getfile(TransformedPairDataset),
        ]
        return supdeps + deps

    def compute(self):
        with open(self.outdir / "stdout.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    self.data_sd = load_dermamnist(
                        root=self.outdir / "medmnist", **self.kwargs
                    )


def load_medmnist_dataset(
    dataset_class, /, mean, std, size, download=True, **kwargs
):
    transform = get_transforms(mean, std, size=size, setting="contrastive")
    transform_lin_train = get_transforms(
        mean, std, size=size, setting="train_linear_classifier"
    )
    transform_none = get_transforms(
        mean, std, size=size, setting="test_linear_classifier"
    )

    kwargs["root"].mkdir(exist_ok=True)
    dataset_train = dataset_class(
        "train",
        download=download,
        **kwargs,
    )
    dataset_val = dataset_class(
        "val",
        download=download,
        **kwargs,
    )
    dataset_test = dataset_class(
        "test",
        download=download,
        **kwargs,
    )
    all_datasets = [dataset_train, dataset_val, dataset_test]
    for dataset in all_datasets:
        dataset.labels = dataset.labels.squeeze()

    dataset_full = ConcatDataset(all_datasets)

    # need to make a dataset that returns two transforms of an image
    # fmt: off
    T = TransformedPairDataset
    test =                T(dataset_test,  transform_none)
    return dict(
        train_contrastive=T(dataset_full,  transform),
        train_linear     =T(dataset_train, transform_lin_train),
        test_linear      =test,
        train_plain      =T(dataset_train, transform_none),
        test_plain       =test,
        full_plain       =T(dataset_full,  transform_none),
    )
    # fmt: on
