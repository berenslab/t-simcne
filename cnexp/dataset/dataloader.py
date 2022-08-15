import torch
from torch.utils.data import DataLoader

from .base import DatasetBase


def make_dataloader(
    dataset, seed=None, batch_size=1024, shuffle=True, num_workers=8, **kwargs
) -> DataLoader:
    if seed is not None:
        gen = torch.Generator()
        gen = gen.manual_seed(seed)
    else:
        gen = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=gen,
        **kwargs,
    )
    return dataloader


class GenericDataLoader(DatasetBase):
    def get_deps(self):
        supdeps = super().get_deps()
        return supdeps + [self.indir / "dataset.pt"]

    def load(self):
        sd = torch.load(self.indir / "dataset.pt")
        self.dataset = sd["dataset"]
        self.state_dict = sd

    def compute(self):
        self.dataloader = make_dataloader(self.dataset, **self.kwargs)

    def save(self):
        self.state_dict.pop("dataloader", None)
        save_data = dict(**self.state_dict, dataloader=self.dataloader)
        self.save_lambda_alt(self.outdir / "dataset.pt", save_data, torch.save)
