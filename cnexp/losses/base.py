import torch
import torch.nn.functional as F

from ..base import ProjectBase


class LossBase(ProjectBase):
    def __init__(self, path, random_state=None, metric="euclidean", **kwargs):
        super().__init__(path, random_state=random_state)
        self.metric = metric
        self.kwargs = kwargs
        # parameters are requires for saving as a state dict.  Not
        # sure how that could influence things down the line so I am
        # only saving the loss object itself for now.
        #
        # self._parameters = kwargs
        # self._buffers = dict()
        # self._modules = dict()
        # self._state_dict_hooks = dict()

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
            # criterion_sd=self.criterion.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)
