import torch
from torch.distributions.laplace import Laplace
from torch.distributions.gamma import Gamma

from sparsecoding.priors.common import Prior


class LSMPrior(Prior):
    """Prior where weights are drawn from i.i.d. from Laplacian scale mixtures.

    The Laplacian scale mixture is defined in:
        Garrigues & Olshausen (2010)
        https://papers.nips.cc/paper/2010/hash/2d6cc4b2d139a53512fb8cbb3086ae2e-Abstract.html
    .

    Conceptually, a Laplacian scale mixture is just a weighted sum of Laplacian distributions
    with different scales.

    In the paper, a Gamma distribution over:
        the inverse of the scale parameter of the Laplacian
    is used,
    as that is the conjugate prior.

    Parameters
    ----------
    dim : int
        Number of weights per sample.
    alpha : float
        Shape or concentration parameter of the Gamma distribution
        over the Laplacian's scale.
    beta : float
        Rate or inverse scale parameter of the Gamma distribution
        over the Laplacian's scale.
    positive_only : bool
        Ensure that the weights are positive by taking the absolute value
        of weights sampled from the Laplacian.
    """

    def __init__(
        self,
        dim: int,
        alpha: float,
        beta: float,
        positive_only: bool = True,
    ):
        if dim < 0:
            raise ValueError(f"`dim` should be nonnegative, got {dim}.")
        if alpha <= 0:
            raise ValueError(f"Must have alpha > 0, got `alpha`={alpha}.")
        if beta <= 0:
            raise ValueError(f"Must have beta > 0, got `beta`={beta}.")

        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.positive_only = positive_only

        self.gamma_distr = Gamma(self.alpha, self.beta)

    @property
    def D(self):
        return self.dim

    def sample(self, num_samples: int):
        N = num_samples

        inverse_lambdas = self.gamma_distr.sample((N, self.D))

        weights = Laplace(
            loc=torch.zeros((N, self.D), dtype=torch.float32),
            scale=1. / inverse_lambdas,
        ).sample()

        if self.positive_only:
            weights = torch.abs(weights)

        return weights

    def log_prob(
        self,
        sample: torch.Tensor,
    ):
        super().check_sample_input(sample)

        log_prob = (
            torch.log(torch.tensor(self.alpha))
            + self.alpha * torch.log(torch.tensor(self.beta))
            - (self.alpha + 1) * torch.log(self.beta + torch.abs(sample))
        )  # [N, D]
        if self.positive_only:
            log_prob[sample < 0.] = -torch.inf
        else:
            log_prob -= torch.log(torch.tensor(2.))

        log_prob = torch.sum(log_prob, dim=1)  # [N]

        return log_prob
