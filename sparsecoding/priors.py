import torch
from torch.distributions.laplace import Laplace

from abc import ABC, abstractmethod


class Prior(ABC):
    """A distribution over weights.

    Parameters
    ----------
    weights_dim : int
        Number of weights for each sample.
    """
    @abstractmethod
    def D(self):
        """
        Number of weights per sample.
        """

    @abstractmethod
    def sample(
        self,
        num_samples: int = 1,
    ):
        """Sample weights from the prior.

        Parameters
        ----------
        num_samples : int, default=1
            Number of samples.

        Returns
        -------
        samples : Tensor, shape [num_samples, self.D]
            Sampled weights.
        """


class SpikeSlabPrior(Prior):
    """Prior where weights are drawn from a "spike-and-slab" distribution.

    The "spike" is at 0 and the "slab" is Laplacian.

    See:
        https://wesselb.github.io/assets/write-ups/Bruinsma,%20Spike%20and%20Slab%20Priors.pdf
    for a good review of the spike-and-slab model.

    Parameters
    ----------
    dim : int
        Number of weights per sample.
    p_spike : float
        The probability of the weight being 0.
    scale : float
        The "scale" of the Laplacian distribution (larger is wider).
    positive_only : bool
        Ensure that the weights are positive by taking the absolute value
        of weights sampled from the Laplacian.
    """

    def __init__(
        self,
        dim: int,
        p_spike: float,
        scale: float,
        positive_only: bool = True,
    ):
        if dim < 0:
            raise ValueError(f"`dim` should be nonnegative, got {dim}.")
        if p_spike < 0 or p_spike > 1:
            raise ValueError(f"Must have 0 <= `p_spike` <= 1, got `p_spike`={p_spike}.")
        if scale <= 0:
            raise ValueError(f"`scale` must be positive, got {scale}.")

        self.dim = dim
        self.p_spike = p_spike
        self.scale = scale
        self.positive_only = positive_only

    @property
    def D(self):
        return self.dim

    def sample(self, num_samples: int):
        N = num_samples

        zero_weights = torch.zeros((N, self.D), dtype=torch.float32)
        slab_weights = Laplace(
            loc=zero_weights,
            scale=torch.full((N, self.D), self.scale, dtype=torch.float32),
        ).sample()  # [N, D]

        if self.positive_only:
            slab_weights = torch.abs(slab_weights)

        spike_over_slab = torch.rand(N, self.D, dtype=torch.float32) < self.p_spike

        weights = torch.where(
            spike_over_slab,
            zero_weights,
            slab_weights,
        )

        return weights


class L0Prior(Prior):
    """Prior with a distribution over the l0-norm of the weights.

    A class of priors where the weights are binary;
    the distribution is over the l0-norm of the weight vector
    (how many weights are active).

    Parameters
    ----------
    prob_distr : Tensor, shape [D], dtype float32
        Probability distribution over the l0-norm of the weights.
    """

    def __init__(
        self,
        prob_distr: torch.Tensor,
    ):
        if prob_distr.dim() != 1:
            raise ValueError(f"`prob_distr` shape must be (D,), got {prob_distr.shape}.")
        if prob_distr.dtype != torch.float32:
            raise ValueError(f"`prob_distr` dtype must be torch.float32, got {prob_distr.dtype}.")
        if not torch.allclose(torch.sum(prob_distr), torch.ones_like(prob_distr)):
            raise ValueError(f"`torch.sum(prob_distr)` must be 1., got {torch.sum(prob_distr)}.")

        self.prob_distr = prob_distr

    @property
    def D(self):
        return self.prob_distr.shape[0]

    def sample(
        self,
        num_samples: int
    ):
        N = num_samples

        num_active_weights = 1 + torch.multinomial(
            input=self.prob_distr,
            num_samples=num_samples,
            replacement=True,
        )  # [N]

        d_idxs = torch.arange(self.D)
        active_idx_mask = (
            d_idxs.reshape(1, self.D)
            < num_active_weights.reshape(N, 1)
        )  # [N, self.D]

        n_idxs = torch.arange(N).reshape(N, 1).expand(N, self.D)  # [N, D]
        # Need to shuffle here so that it's not always the first weights that are active.
        shuffled_d_idxs = [torch.randperm(self.D) for _ in range(N)]
        shuffled_d_idxs = torch.stack(shuffled_d_idxs, dim=0)  # [N, D]

        # [num_active_weights], [num_active_weights]
        active_weight_idxs = n_idxs[active_idx_mask], shuffled_d_idxs[active_idx_mask]

        weights = torch.zeros((N, self.D), dtype=torch.float32)
        weights[active_weight_idxs] += 1.

        return weights
