import torch

from ..base import ProjectBase


class LossBase(ProjectBase):
    def __init__(self, path, random_state=None, metric="euclidean", **kwargs):
        super().__init__(path, random_state=random_state)
        self.metric = metric
        self.kwargs = kwargs

    def get_deps(self):
        return []

    def load(self):
        pass

    def save(self):
        save_data = dict(
            criterion=self.criterion,
            criterion_sd=self.criterion.state_dict(),
        )
        self.save_lambda_alt(
            self.outdir / "criterion.pt", save_data, torch.save
        )
