import torch
from torch import optim

from .base import ProjectBase


class OptimBase(ProjectBase):
    def __init__(self, path, random_state=None, lr="lin-bs", **kwargs):
        super().__init__(path, random_state=random_state)
        # kwargs.setdefault("momentum", 0.9)
        # kwargs.setdefault("weight_decay", 5e-4)
        self.lr = lr
        self.kwargs = kwargs

    def get_deps(self):
        if self.lr == "lin-bs":
            extra_dep = [self.indir / "dataset.pt"]
        else:
            extra_dep = []

        return [self.indir / "model.pt"] + extra_dep

    def load(self):
        self.state_dict = torch.load(self.indir / "model.pt")
        self.model = self.state_dict["model"]

        if self.lr == "lin-bs":
            loader = torch.load(self.indir / "dataset.pt")[
                "train_contrastive_loader"
            ]
            self.batch_size = loader.batch_size
        else:
            self.batch_size = None

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


def lr_from_batchsize(batch_size: int, /, mode="lin-bs") -> float:
    if mode == "lin-bs":
        lr = 0.03 * batch_size / 256
    elif mode == "sqrt-bs":
        lr = 0.075 * batch_size**0.5
    else:
        raise ValueError(f"Unknown mode for calculating the lr ({mode = !r})")

    return lr


def make_sgd(
    model, lr=0.12, momentum=0.9, weight_decay=5e-4, batch_size=None, **kwargs
):
    if batch_size is not None and isinstance(lr, str):
        lr = lr_from_batchsize(batch_size)
    elif isinstance(lr, (int, float)):
        lr = lr
    else:
        raise ValueError(f"No valid {lr = } or {batch_size = } passed")

    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )


class SGD(OptimBase):
    def compute(self):
        self.opt = make_sgd(
            self.model, lr=self.lr, batch_size=self.batch_size, **self.kwargs
        )


def make_adam(
    model, lr=0.12, momentum=0.9, weight_decay=5e-4, batch_size=None, **kwargs
):
    if batch_size is not None and isinstance(lr, str):
        lr = lr_from_batchsize(batch_size)
    elif isinstance(lr, (int, float)):
        lr = lr
    else:
        raise ValueError(f"No valid {lr = } or {batch_size = } passed")

    return optim.Adam(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs,
    )


class Adam(OptimBase):
    def compute(self):
        self.opt = make_adam(
            self.model, lr=self.lr, batch_size=self.batch_size, **self.kwargs
        )
