from typing import Union

import torch
from torch import nn

from .simclr_like import SimCLRModel, make_projection_head


def mutate_model(
    model: nn.Module,
    change: str = "nothing",
    freeze: Union[str, bool, None] = None,
    proj_head="mlp",
    out_dim: int = 2,
    hidden_dim=None,
    last_lin_std: float = 1.0,
    # **kwargs,
):
    """mutate a given model for further finetuning.

    This function does two things:
        1. It changes whether the model requires a gradient (via `freeze`).
        2. It swaps out a part of the model (currently only in the proj. head).
           This is controlled via the parameter `change`.

    By default the function does not do anything and will just return the model
    as is.
    """
    if freeze is not None:
        if freeze == "backbone":
            model.backbone.requires_grad_(False)
            model.projection_head.requires_grad_(True)
        elif freeze == "thaw_lastlin":
            # unfreeze the last linear layer.  Useful for finetuning
            # the same model on a different loss.
            model.requires_grad_(False)
            model.projection_head.layers[-1].requires_grad_(True)
        else:
            model.requires_grad_(not freeze)

    if change == "lastlin":
        last_layer = model.projection_head.layers[-1]
        orig_out_dim, dim = last_layer.weight.shape

        if orig_out_dim != out_dim:
            # swap out the last linear layer of the projection head
            lin = nn.Linear(dim, out_dim)
            nn.init.normal_(lin.weight, std=last_lin_std)
            model.projection_head.layers[-1] = lin
        else:
            if last_lin_std != 1:
                lin = model.projection_head.layers[-1]
                lin.weight = torch.nn.Parameter(lin.weight * last_lin_std)
                lin.bias = torch.nn.Parameter(lin.bias * last_lin_std)

    elif change == "proj_head":
        # swap out the entire projection head
        in_dim = model.backbone_dim
        if hidden_dim is None:
            # try to infer the dim from the previous projection head
            hidden_dim = model.projection_head.layers[0].weight.size(0)
        model.projection_head = make_projection_head(
            proj_head, in_dim, hidden_dim, out_dim
        )

    elif change == "nothing":
        pass

    else:
        raise ValueError(
            f"Requested to {change = !r}, but I don't know what to do"
        )

    return model


class FinetuneSimCLRModel(SimCLRModel):
    def get_deps(self):
        supdeps = super().get_deps()
        return supdeps + [self.indir / "model.pt"]

    def load(self):
        self.state_dict = torch.load(
            self.indir / "model.pt", map_location="cpu"
        )
        self.model = self.state_dict["model"]

    def compute(self):
        self.model = mutate_model(self.model, **self.kwargs)

    def save(self):
        # remove old values that might be present
        self.state_dict.pop("model", None)
        self.state_dict.pop("model_sd", None)

        save_data = dict(
            **self.state_dict,
            model=self.model,
            model_sd=self.model.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)


class ReadoutModel(FinetuneSimCLRModel):
    """A model that is prepared for the linear readout stage.

    Corresponds to the finetuned model with the following parameters:
    ftmodel:change=proj_head:freeze=1:proj_head=linear and will
    attempt to infer the out_dim, unless `out_dim` is passed
    explicitly.

    """

    def __init__(
        self,
        path,
        change="proj_head",
        freeze="backbone",
        proj_head="linear",
        out_dim="infer",
        **kwargs,
    ):
        super().__init__(
            path,
            change=change,
            freeze=freeze,
            proj_head=proj_head,
            out_dim=out_dim,
            **kwargs,
        )

    def get_deps(self):
        supdeps = super().get_deps()
        infer_outdim = self.kwargs["out_dim"] == "infer"

        deps = (
            (supdeps + [self.indir / "dataset.pt"])
            if infer_outdim
            else supdeps
        )
        return deps

    def load(self):
        super().load()

        infer_outdim = self.kwargs["out_dim"] == "infer"
        if infer_outdim:
            data_dict = torch.load(self.indir / "dataset.pt")
            dataset = data_dict["train_contrastive"]
            try:
                self.kwargs["out_dim"] = len(dataset.classes)
            except AttributeError:
                raise RuntimeError(
                    "Cannot infer outdim from preceding dataset "
                    f"and outdim was set to {self.kwargs['out_dim']}"
                )
