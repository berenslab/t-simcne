import torch
import torch.nn.functional as F
from torch import nn


class InfoNCEDot(nn.Module):
    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # if self.normalize:
        a = F.normalize(a)
        b = F.normalize(b)
        # else:
        #     a = F.normalize(a)
        #     b = F.normalize(b, dim=0)

        cos_aa = a @ a.T / self.temperature
        # cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        # cos_bb.masked_fill_(self_mask, float("-inf"))
        # logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_2  # logsumexp_1 +

        loss = -(tempered_alignment - raw_uniformity)  # / 2
        return loss
