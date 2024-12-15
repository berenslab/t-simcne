from typing import Union

import torch
from torch import nn

from .simclr_like import make_projection_head


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
