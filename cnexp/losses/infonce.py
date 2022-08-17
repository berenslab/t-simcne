import torch
import torch.nn.functional as F
from torch import nn

from .base import LossBase


class InfoNCECosine(nn.Module):
    def __init__(self, temperature=0.5):
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

        # cdist does not necessarily return 0 for the self-distance,
        # but this is not accounted for here.  This should be fine™
        sim_aa = 1 / (1 + torch.cdist(a, a))
        sim_bb = 1 / (1 + torch.cdist(b, b))
        sim_ab = 1 / (1 + torch.cdist(a, b))

        tempered_alignment = sim_ab.trace() / batch_size

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCELoss(LossBase):
    def compute(self):
        metric = self.metric
        if metric == "cosine":
            self.criterion = InfoNCECosine(**self.kwargs)
        elif metric == "euclidean":
            self.criterion = InfoNCEEuclidean(**self.kwargs)
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")
