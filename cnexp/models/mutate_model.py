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
        # elif freeze == "thaw_llin":
        #     # unfreeze the last linear layer.  This shouldn't really
        #     # be necessary, since introducing the new linear layer
        #     # with the `change='lastlin'` parameter will already
        #     # require a gradient.
        #     model.requires_grad_(False)
        #     model.projection_head.layers[-1].requires_grad_(True)
        else:
            model.requires_grad_(not freeze)

    if change == "lastlin":
        # swap out the last linear layer of the projection head
        last_layer = model.projection_head.layers[-1]
        dim = last_layer.weight.size(1)
        model.projection_head.layers[-1] = nn.Linear(dim, out_dim)

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
        self.state_dict = torch.load(self.indir / "model.pt")
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
    ftmodel:change=proj_head:freeze=1:proj_head=linear:out_dim=10.  So
    this will actually only set some parameters for the super class.

    """

    def __init__(
        self,
        path,
        change="proj_head",
        freeze="backbone",
        proj_head="linear",
        **kwargs,
    ):
        super().__init__(
            path, change=change, freeze=freeze, proj_head=proj_head, **kwargs
        )