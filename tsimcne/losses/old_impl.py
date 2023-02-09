import torch
from torch.nn.functional import normalize

from .base import LossBase


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        negative_samples=2048,
        temperature=0.5,
        loss_mode="infonce",
        metric="euclidean",
        eps=1.0,
        noise_in_estimator=1.0,
        clamp_lo=float("-inf"),
        clamp_hi=float("inf"),
        seed=0,
        loss_aggregation="mean",
    ):
        super(ContrastiveLoss, self).__init__()
        self.negative_samples = negative_samples
        self.temperature = temperature
        self.loss_mode = loss_mode
        self.metric = metric
        self.noise_in_estimator = noise_in_estimator
        self.eps = eps
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.clamp = self.clamp_lo, self.clamp_hi
        self.seed = seed
        torch.manual_seed(self.seed)
        self.neigh_inds = None
        self.loss_aggregation = loss_aggregation

        if self.loss_mode == "nce":
            self.log_Z = torch.tensor(0.0)
            self.log_Z = torch.nn.Parameter(self.log_Z, requires_grad=True)

    def forward(
        self,
        features,
        backbone_features=None,
        labels=None,
        force_resample=False,
    ):
        """Compute loss for model. SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [2 * bsz, n_views, ...].
            force_resample: Whether the negative samples should be forcefully resampled.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0] // 2
        b = batch_size

        # We can at most sample this many samples from the batch.
        # `b` can be lower than `self.negative_samples` in the last batch.
        negative_samples = min(self.negative_samples, 2 * b - 1)

        if force_resample or self.neigh_inds is None:
            neigh_inds = make_neighbor_indices(
                batch_size, negative_samples, device=features.device
            )
            self.neigh_inds = neigh_inds
        # untested logic to accomodate for last batch
        elif self.neigh_inds.shape[0] != batch_size:
            neigh_inds = make_neighbor_indices(
                batch_size, negative_samples, device=features.device
            )
            # don't save this one
        else:
            neigh_inds = self.neigh_inds
        neighbors = features[neigh_inds, :]

        # `neigh_mask` indicates which samples feel attractive force
        # and which ones repel each other
        neigh_mask = torch.ones_like(neigh_inds, dtype=torch.bool)
        neigh_mask[:, 0] = False

        origs = features[:b]

        # compute probits
        if self.metric == "euclidean":
            dists = ((origs[:, None] - neighbors) ** 2).sum(axis=2)
            # Cauchy affinities
            probits = torch.div(1, self.eps + dists)
        elif self.metric == "cosine":
            o = normalize(origs.unsqueeze(1), dim=2)
            n = normalize(neighbors.transpose(1, 2), dim=1)
            logits = torch.bmm(o, n).squeeze() / self.temperature
            # logits_max, _ = logits.max(dim=1, keepdim=True)
            # logits -= logits_max.detach()
            # logits -= logits.max().detach()
            probits = torch.exp(logits)
        else:
            raise ValueError(f"Unknown metric “{self.metric}”")

        # compute loss
        if self.loss_mode == "nce":
            self.log_Z.to(features.device)

            # for proper nce it should be negative_samples *
            # p_noise. But for uniform noise distribution we would
            # need the size of the dataset here. Also, we do not use a
            # uniform noise distribution as we sample negative samples
            # from the batch.

            if self.metric == "euclidean":
                # estimator is (cauchy / Z) / ( cauchy / Z +
                # neg_samples)). For numerical stability rewrite to 1
                # / ( 1 + (d**2 + eps) * Z * m)
                estimator = 1 / (
                    1
                    + (dists + self.eps)
                    * torch.exp(self.log_Z)
                    * negative_samples
                )
            else:
                probits = probits / torch.exp(self.log_Z)
                estimator = probits / (probits + negative_samples)

            loss = -(~neigh_mask * torch.log(estimator.clamp(*self.clamp))) - (
                neigh_mask * torch.log((1 - estimator).clamp(*self.clamp))
            )
        elif self.loss_mode == "neg_sample":
            if self.metric == "euclidean":
                # estimator rewritten for numerical stability as for nce
                estimator = 1 / (
                    1 + self.noise_in_estimator * (dists + self.eps)
                )
            else:
                estimator = probits / (probits + self.noise_in_estimator)

            loss = -(~neigh_mask * torch.log(estimator.clamp(*self.clamp))) - (
                neigh_mask * torch.log((1 - estimator).clamp(*self.clamp))
            )

        elif self.loss_mode == "umap":
            # cross entropy parametric umap loss
            loss = -(~neigh_mask * torch.log(probits.clamp(*self.clamp))) - (
                neigh_mask * torch.log((1 - probits).clamp(*self.clamp))
            )
        elif self.loss_mode == "infonce":
            # loss from e.g. sohn et al 2016, includes pos similarity
            # in denominator
            loss = -(
                (torch.log(probits.clamp(*self.clamp)[~neigh_mask]))
                - torch.log(probits.clamp(*self.clamp).sum(axis=1))
            )
        elif self.loss_mode == "infonce_alt":
            # loss simclr
            loss = -(
                (torch.log(probits.clamp(*self.clamp)[~neigh_mask]))
                - torch.log(
                    (neigh_mask * probits.clamp(*self.clamp)).sum(axis=1)
                )
            )
        else:
            raise ValueError(f"Unknown loss_mode “{self.loss_mode}”")

        # aggregate loss over batch
        if self.loss_aggregation == "sum":
            loss = loss.sum()
        else:
            loss = loss.mean()

        return loss


def make_neighbor_indices(batch_size, negative_samples, device=None):
    """
    Selects neighbor indices
    :param batch_size: int Batch size
    :param negative_samples: int Number of negative samples
    :param device: torch.device Device of the model
    :return: torch.tensor Neighbor indices
    :rtype:
    """
    b = batch_size

    if negative_samples < 2 * b - 1:
        # uniform probability for all points in the minibatch,
        # we sample points for repulsion randomly
        neg_inds = torch.randint(
            0, 2 * b - 1, (b, negative_samples), device=device
        )
        neg_inds += (torch.arange(1, b + 1, device=device) - 2 * b)[:, None]
    else:
        # full batch repulsion
        all_inds1 = torch.repeat_interleave(
            torch.arange(b, device=device)[None, :], b, dim=0
        )
        not_self = ~torch.eye(b, dtype=bool, device=device)
        neg_inds1 = all_inds1[not_self].reshape(b, b - 1)

        all_inds2 = torch.repeat_interleave(
            torch.arange(b, 2 * b, device=device)[None, :], b, dim=0
        )
        neg_inds2 = all_inds2[not_self].reshape(b, b - 1)
        neg_inds = torch.hstack((neg_inds1, neg_inds2))

    # now add transformed explicitly
    neigh_inds = torch.hstack(
        (torch.arange(b, 2 * b, device=device)[:, None], neg_inds)
    )

    return neigh_inds


class SlowContrastiveLoss(LossBase):
    def get_deps(self):
        supdeps = super().get_deps()
        # the default return value for kwargs.get needs to match the
        # "loss_mode" kwarg in the ContrastiveLoss class
        if self.kwargs.get("loss_mode", "infonce") == "nce":
            deps = [self.indir / "model.pt"]
        else:
            deps = []
        return supdeps + deps

    def load(self):
        if self.kwargs.get("loss_mode", "infonce") == "nce":
            self.model_sd = torch.load(self.indir / "model.pt")

    def compute(self):
        self.seed = self.random_state.integers(2**63 - 1)
        self.criterion = ContrastiveLoss(
            metric=self.metric, seed=self.seed, **self.kwargs
        )

        # Add the normalization constant to the optimizer
        if self.kwargs.get("loss_mode", "infonce") == "nce":
            self.opt = self.model_sd["opt"]
            self.opt.add_param_group(dict(params=self.criterion.log_Z))

    def save(self):
        super().save()

        if self.kwargs.get("loss_mode", "infonce") == "nce":
            self.model_sd["opt"] = self.opt
            self.model_sd["opt_sd"] = self.opt.state_dict()
            self.save_lambda_alt(
                self.outdir / "model.pt", self.model_sd, torch.save
            )
