import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

from .base import ProjectBase


class LRScheduleBase(ProjectBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        # if we want to pass some defaults to all that derive from here
        # kwargs.setdefault("momentum", 0.9)
        self.kwargs = kwargs

    def get_deps(self):
        return [self.indir / "model.pt"]

    def load(self):
        self.state_dict = torch.load(self.indir / "model.pt")
        if "opt" not in self.state_dict.keys():
            raise ValueError(
                f"Optimizer not found in {self.indir / 'model.pt'}, "
                "cannot create an LR scheduler"
            )
        self.opt = self.state_dict["opt"]

    def save(self):
        # remove old values that might be present
        self.state_dict.pop("lrsched", None)
        self.state_dict.pop("lrsched_sd", None)

        save_data = dict(
            **self.state_dict,
            lrsched=self.lr,
            lrsched_sd=self.lr.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)


class CosineAnnealingSchedule(_LRScheduler):
    """Cosine annealing with warmup."""

    def __init__(
        self, opt, final_lr=0, n_epochs=1000, warmup_epochs=10, warmup_lr=0
    ):
        self.opt = opt
        self.optimizer = self.opt
        self.base_lr = base_lr = opt.defaults["lr"]
        self.final_lr = final_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        # increase the number by one since we initialize the optimizer
        # before the first step (so the lr is set to 0 in the case of
        # warmups).  So we start counting at 1, basically.
        decay_epochs = 1 + n_epochs - warmup_epochs
        self.decay_epochs = decay_epochs

        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        decay_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs)
        )
        self.lr_schedule = np.hstack((warmup_schedule, decay_schedule))

        self._last_lr = self.lr_schedule[0]
        self.cur_epoch = 0

        self.init_opt()

    def init_opt(self):
        self.step()
        # self.set_epoch(0)

    def get_lr(self):
        return self.lr_schedule[self.cur_epoch]

    def step(self):
        for param_group in self.opt.param_groups:
            lr = param_group["lr"] = self.get_lr()

        self.cur_epoch += 1
        self._last_lr = lr
        return lr

    def set_epoch(self, epoch):
        self.cur_epoch = epoch


class CosineAnnealing(LRScheduleBase):
    def compute(self):
        self.lr = CosineAnnealingSchedule(self.opt, **self.kwargs)


class LinearAnnealingSchedule(CosineAnnealingSchedule):
    def __init__(
        self, opt, final_lr=0, n_epochs=1000, warmup_epochs=10, warmup_lr=0
    ):
        self.opt = opt
        self.base_lr = base_lr = opt.defaults["lr"]
        self.final_lr = final_lr
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        # increase the number by one since we initialize the optimizer
        # before the first step (so the lr is set to 0 in the case of
        # warmups).  So we start counting at 1, basically.
        decay_epochs = 1 + n_epochs - warmup_epochs
        self.decay_epochs = decay_epochs

        warmup_schedule = np.linspace(warmup_lr, base_lr, warmup_epochs)
        decay_schedule = np.linspace(base_lr, final_lr, decay_epochs)
        self.lr_schedule = np.hstack((warmup_schedule, decay_schedule))

        self._last_lr = self.lr_schedule[0]
        self.cur_epoch = 0

        self.init_opt()


class LinearAnnealing(LRScheduleBase):
    def compute(self):
        self.lr = LinearAnnealingSchedule(self.opt, **self.kwargs)
