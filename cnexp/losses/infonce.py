import inspect

import torch
import torch.nn.functional as F
from torch import nn

from .base import LossBase


class InfoNCECosine(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCEEuclidean(nn.Module):
    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (1 + torch.cdist(a, a) ** 2)
        sim_bb = 1 / (1 + torch.cdist(b, b) ** 2)
        sim_ab = 1 / (1 + torch.cdist(a, b) ** 2)

        tempered_alignment = torch.diagonal_copy(sim_ab).log_().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCELoss(LossBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

        metric = self.metric
        if metric == "cosine":
            self.cls = InfoNCECosine
        elif metric == "euclidean":
            self.cls = InfoNCEEuclidean
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

    def get_deps(self):
        supdeps = super().get_deps()
        return [inspect.getfile(self.cls)] + supdeps

    def compute(self):
        self.criterion = self.cls(**self.kwargs)
