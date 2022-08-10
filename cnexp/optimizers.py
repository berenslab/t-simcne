import torch
from torch import optim

from .base import ProjectBase


class OptimBase(ProjectBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        # kwargs.setdefault("momentum", 0.9)
        # kwargs.setdefault("weight_decay", 5e-4)
        self.kwargs = kwargs

    def get_deps(self):
        return [self.indir / "model.pt"]

    def load(self):
        self.state_dict = torch.load(self.indir / "model.pt")
        self.model = self.state_dict["model"]

    def save(self):
        # remove old values that might be present
        self.state_dict.pop("opt", None)
        self.state_dict.pop("opt_sd", None)

        save_data = dict(
            **self.state_dict,
            opt=self.opt,
            opt_sd=self.opt.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)


def make_sgd(model, lr=0.12, momentum=0.9, weight_decay=5e-4, **kwargs):
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )


class SGD(OptimBase):
    def compute(self):
        self.opt = make_sgd(self.model, **self.kwargs)


def make_adam(model, lr=0.12, momentum=0.9, weight_decay=5e-4, **kwargs):
    return optim.Adam(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )


class Adam(OptimBase):
    def compute(self):
        self.opt = make_adam(self.model, **self.kwargs)
