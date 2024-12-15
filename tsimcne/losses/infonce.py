import inspect

import torch
import torch.nn.functional as F
from torch import nn

from .base import LossBase


class InfoNCECosine(nn.Module):
    def __init__(
        self,
        temperature: float = 0.5,
        reg_coef: float = 0,
        reg_radius: float = 200,
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

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

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return dict(
            loss=loss,
            ta=-tempered_alignment,
            ru=raw_uniformity / 2,
        )


class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return dict(
            loss=loss,
            ta=-tempered_alignment,
            ru=raw_uniformity / 2,
        )


class InfoNCELoss(LossBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

        metric = self.metric
        if metric == "cosine":
            self.cls = InfoNCECosine
        elif metric == "euclidean":  # actually Cauchy
            self.cls = InfoNCECauchy
        elif metric == "gauss":
            self.cls = InfoNCEGaussian
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

    def get_deps(self):
        supdeps = super().get_deps()
        return [inspect.getfile(self.cls)] + supdeps

    def compute(self):
        self.criterion = self.cls(**self.kwargs)


class InfoNCET(InfoNCEGaussian):
    def __init__(self, dof=None, temperature=1, **kwargs):
        super().__init__(**kwargs)
        self.dof = dof
        self.temperature = temperature

    def forward(self, features):
        batch_size = features.size(0) // 2

        features = features.float()
        a = features[:batch_size]
        b = features[batch_size:]

        d_aa = torch.cdist(a, a).square() * self.temperature
        d_bb = torch.cdist(b, b).square() * self.temperature
        d_ab = torch.cdist(a, b).square() * self.temperature

        if self.dof is None:
            dof = max(2, features.size(1) // 10)
        else:
            dof = self.dof

        # log-similarity, student t-kernel
        sim_aa = (d_aa / (0.5 * dof)).log1p() * (-0.5 * (dof + 1))
        sim_bb = (d_bb / (0.5 * dof)).log1p() * (-0.5 * (dof + 1))
        sim_ab = (d_ab / (0.5 * dof)).log1p() * (-0.5 * (dof + 1))

        tempered_alignment = sim_ab.diagonal().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(self.exaggeration * tempered_alignment - raw_uniformity / 2)
        return dict(
            loss=loss,
            # left=sqa.mean(),
            # inner=iab.mean(),
            # right=sqb.mean(),
            ta=-tempered_alignment,
            ru=raw_uniformity / 2,
        )


class InfoNCECauchy(InfoNCET):
    def __init__(self, temperature: float = 1):
        super().__init__(dof=1, temperature=temperature)
