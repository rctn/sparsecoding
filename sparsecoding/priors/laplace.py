import torch
from torch.distributions.laplace import Laplace

from sparsecoding.priors.common import Prior


class LaplacePrior(Prior):
    """Prior corresponding to a Laplacian distribution.

    Parameters
    ----------
    dim : int
        Number of weights per sample.
    scale : float
        The "scale" of the Laplacian distribution (larger is wider).
    positive_only : bool
        Ensure that the weights are positive by taking the absolute value
        of weights sampled from the Laplacian.
    """

    def __init__(
        self,
        dim: int,
        scale: float,
        positive_only: bool = True,
    ):
        if dim < 0:
            raise ValueError(f"`dim` should be nonnegative, got {dim}.")
        if scale <= 0:
            raise ValueError(f"`scale` must be positive, got {scale}.")

        self.dim = dim
        self.scale = scale
        self.positive_only = positive_only

        self.distr = Laplace(loc=torch.tensor(0.), scale=torch.tensor(self.scale))

    @property
    def D(self):
        return self.dim

    def sample(self, num_samples: int):
        weights = self.distr.rsample((num_samples, self.D))
        if self.positive_only:
            weights = torch.abs(weights)
        return weights

    def log_prob(
        self,
        sample: torch.Tensor,
    ):
        super().check_sample_input(sample)

        log_prob = self.distr.log_prob(sample)
        if self.positive_only:
            log_prob += torch.log(torch.tensor(2.))
            log_prob[sample < 0.] = -torch.inf
        log_prob = torch.sum(log_prob, dim=1)  # [N]

        return log_prob
