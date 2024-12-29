import torch
import torch.nn.functional as F
from torch import nn


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
        return dict(
            loss=loss,
            ta=-tempered_alignment,
            ru=raw_uniformity / 2,
        )


class InfoNCEGaussian(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

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


class InfoNCET(InfoNCEGaussian):
    def __init__(self, dof=None, temperature=1):
        super().__init__(temperature=temperature)
        self.dof = dof

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

        loss = -(tempered_alignment - raw_uniformity / 2)
        return dict(
            loss=loss,
            ta=-tempered_alignment,
            ru=raw_uniformity / 2,
        )


class InfoNCECauchy(InfoNCET):
    def __init__(self, temperature: float = 1):
        super().__init__(dof=1, temperature=temperature)


class CauchyTemp(InfoNCECauchy):
    def __init__(self, temperature: float = 1.0):
        super().__init__(temperature=temperature)

        self.logtemp = torch.nn.Parameter(torch.tensor(self.temperature).log())

    def __call__(self, features):
        self.temperature = self.logtemp.exp()
        lossdict = super().__call__(features)
        lossdict.update(logtemp=self.logtemp.item())
        return lossdict


class CosineTemp(InfoNCECosine):
    def __init__(self, temperature: float = 0.5):
        super().__init__(temperature=temperature)

        self.logtemp = torch.nn.Parameter(torch.tensor(self.temperature).log())

    def __call__(self, features):
        self.temperature = self.logtemp.exp()
        lossdict = super().__call__(features)
        lossdict.update(logtemp=self.logtemp.item())
        return lossdict
