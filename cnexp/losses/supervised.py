import torch
import torch.nn.functional as F

from .base import LossBase


class CrossEntropy(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features are unused
        batch_size = features.size(0) // 2
        return F.cross_entropy(
            features[:batch_size], labels.to(features.device), **self.kwargs
        )


class CELoss(LossBase):
    def compute(self):
        self.criterion = CrossEntropy(**self.kwargs)
