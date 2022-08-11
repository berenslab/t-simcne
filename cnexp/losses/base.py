import torch
import torch.nn.functional as F

from ..base import ProjectBase


class LossBase(ProjectBase):
    def __init__(self, path, random_state=None, metric="euclidean", **kwargs):
        super().__init__(path, random_state=random_state)
        self.metric = metric
        self.kwargs = kwargs
    def get_deps(self):
        return []

    def load(self):
        if (self.indir / "model.pt").exists():
            self.state_dict = torch.load(self.indir / "model.pt")
        else:
            self.state_dict = dict()

    def save(self):
        self.state_dict.pop("criterion", None)
        self.state_dict.pop("criterion_sd", None)

        save_data = dict(
            **self.state_dict,
            criterion=self.criterion,
            criterion_sd=self.criterion.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)
