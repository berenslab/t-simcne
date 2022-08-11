import torch
import torch.nn.functional as F

from ..base import ProjectBase


class LossBase(ProjectBase):
    def __init__(self, path, random_state=None, **kwargs):
        super().__init__(path, random_state=random_state)
        self.kwargs = kwargs
    def get_deps(self):
        return []

    def load(self):
        self.state_dict = torch.load(self.indir / "model.pt")
        self.model = self.state_dict["model"]

    def save(self):
        self.state_dict.pop("criterion", None)
        self.state_dict.pop("criterion_sd", None)

        save_data = dict(
            **self.state_dict,
            criterion=self.criterion,
            criterion_sd=self.criterion.state_dict(),
        )
        self.save_lambda_alt(self.outdir / "model.pt", save_data, torch.save)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    """Calculates the cosine similarity for the contrastive loss.

    Parameters
    ----------
    a : torch.Tensor

        has the shape `b ⨯ d` where `b` is the batch size and `d` is
        the latent dimension (usually in Z in the SimCLR setup).
    b : torch.Tensor
        has the shape `b ⨯ m ⨯ d` where `m` is the number of negative
        samples.  If the full-batch repulsion (normal SimCLR setup) is
        used then this parameter should be `2b`.

    Returns
    -------
    torch.Tensor
        a new tensor with the shape `b ⨯ m` holding the batchwise
        similarities.

    """
    a_ = F.normalize(a.unsqueeze(1), dim=2)
    b_ = F.normalize(b, dim=2).transpose(1, 2)
    # the matrix multiplication will do a batch matrix multiplication
    # in this case; it's equivalent to torch.bmm
    return (a_ @ b_).squeeze()


def euclidean_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    _sqdiff=torch.nn.MSELoss(reduction="none"),
):
    """Calculates the Euclidean similarity for the contrastive loss.

    Parameters
    ----------
    a : torch.Tensor
        has the shape `b ⨯ d` where `b` is the batch size and `d` is
        the latent dimension (usually in Z in the SimCLR setup).
    b : torch.Tensor
        has the shape `b ⨯ m ⨯ d` where `m` is the number of negative
        samples.  If the full-batch repulsion (normal SimCLR setup) is
        used then this parameter should be `2b`.
    _sqdiff : torch.Module
        do not alter this parameter or supply another one in its
        place.  It is used to calculate the squared difference between
        the samples in `a` and `b`.

    Returns
    -------
    torch.Tensor
        a new tensor with the shape `b ⨯ m` holding the batchwise
        similarities.

    """
    a_ = a.unsqueeze(1).expand_as(b)
    dists = _sqdiff(a_, b).sum(axis=2)
    return 1 / (1 + dists)


#
# Irrelevant code, mostly for documentation.
#
def __benchmark():
    """The rationale behind the similarity function implementation.

    The functions have been timed with the `%timeit` ipython magic
    command, not with the manual setup of `timeit.timeit`.  If you
    want to test this for youself, it is recommended to paste the
    functions into ipython/jupyter and run %timeit there:

        %timeit eucsim1(a, b)

    This function shows a small benchmark of some implementations,
    where the most performant one has been used in `cosine_similarity`
    and `euclidean_similarity`.

    """
    import timeit

    def eucsim1(a, b):
        """Naive impl. from old project."""
        dists = ((a[:, None] - b) ** 2).sum(axis=2)
        return 1 / (1 + dists)

    def eucsim2(a: torch.Tensor, b: torch.Tensor):
        """functional"""
        a_ = a.unsqueeze(1).expand_as(b)
        dists = torch.nn.functional.mse_loss(a_, b, reduction="none").sum(
            axis=2
        )
        return 1 / (1 + dists)

    def cosim1(a, b):
        """Using torch.functional."""
        return F.cosine_similarity(a.unsqueeze(1), b, dim=2)

    # average tensor for the loss function
    a = torch.rand(1024, 128)
    b = torch.rand(1024, 2048, 128)
    if torch.cuda.is_available():
        a = a.cuda()
        b = b.cuda()

    # define them here so that they are part of locals()
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
        a_ = F.normalize(a.unsqueeze(1), dim=2)
        b_ = F.normalize(b, dim=2).transpose(1, 2)
        # the matrix multiplication will do a batch matrix multiplication
        # in this case; it's equivalent to torch.bmm
        return (a_ @ b_).squeeze()

    def euclidean_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
        _sqdiff=torch.nn.MSELoss(reduction="none"),
    ):
        """Using torch.nn.MSELoss. Not much difference to the real one"""
        a_ = a.unsqueeze(1).expand_as(b)
        dists = _sqdiff(a_, b).sum(axis=2)
        return 1 / (1 + dists)

    assert torch.allclose(euclidean_similarity(a, b), eucsim2(a, b))
    assert torch.allclose(eucsim1(a, b), eucsim2(a, b))
    assert torch.allclose(cosine_similarity(a, b), cosim1(a, b))

    for f in [
        eucsim1,
        euclidean_similarity,
        eucsim2,
        cosim1,
        cosine_similarity,
    ]:
        ts = timeit.repeat(
            f"{f.__name__}(a, b)", repeat=7, number=100, globals=locals()
        )
        t = torch.tensor(ts)
        print(f"{f.__name__[:10]:10s}: {t.mean():.3f} ± {t.std():.3f} secs")

    # for some reason this benchmark claims that `eucsim` is faster
    # than `euclidean_similarity`, but I haven't managed to reproduce
    # this in an interactive session. Not sure why it does not really
    # work, but in the end it doesn't make a difference between the
    # two:

    # In [198]: %timeit euclidean_similarity(a, b)
    #      ...: %timeit eucsim2(a, b)
    # 4.74 ms ± 92.3 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # 4.74 ms ± 123 ns per loop (mean ± std. dev. of 7 runs, 100 loops each)
