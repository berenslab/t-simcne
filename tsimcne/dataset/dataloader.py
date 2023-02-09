import os

import torch
from torch.utils.data import DataLoader

from .base import DatasetBase


def make_dataloader(
    dataset,
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = -1,
    **kwargs,
) -> DataLoader:

    # interpret numbers lower than 0 like in sklearn (-1 means all
    # available cores).  This is not supported by pytorch, hence we
    # need to retrieve the cpu count manually.
    if num_workers < 0:
        # cap concurrent workers.  At some point the performance
        # actually degrades.
        num_workers = min(num_workers + os.cpu_count() + 1, 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
    return dataloader


def make_dataloaders(datasets: dict, **kwargs) -> dict:
    """Takes in a dict of Datasets and returns a dict of DataLoaders.

    The keys in the return dict will be the keys of the original dict
    with "_loader" appended to it.

    Parameters
    ----------
    datasets : dict of torch.utils.data.Dataset
        Datasets that will be wrapped with a DataLoader.
    **kwargs : any
        Will be passed on to `make_dataloader`.

    Returns
    -------
    dict of torch.utils.data.DataLoader
    """
    keys = [(f"{key}_loader", key) for key in datasets.keys()]
    loaders = dict()

    for l_key, data_key in keys:

        if data_key.startswith("test_") or data_key == "full_plain":
            shuffle = False
        else:
            shuffle = True

        loaders[l_key] = make_dataloader(
            datasets[data_key], shuffle=shuffle, **kwargs
        )

    return loaders


class GenericDataLoader(DatasetBase):
    def get_deps(self):
        supdeps = super().get_deps()
        return supdeps + [self.indir / "dataset.pt"]

    def load(self):
        self.state_dict = torch.load(self.indir / "dataset.pt")

    def compute(self):
        self.dataloader_dict = make_dataloaders(self.state_dict, **self.kwargs)

    def save(self):
        save_data = dict(**self.state_dict, **self.dataloader_dict)
        self.save_lambda_alt(self.outdir / "dataset.pt", save_data, torch.save)
