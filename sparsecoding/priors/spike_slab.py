import torch
from torch.distributions.laplace import Laplace

from sparsecoding.priors.common import Prior


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
