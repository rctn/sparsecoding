import torch

from sparsecoding.priors.common import Prior


class SpikeSlabPrior(Prior):
    """Prior where weights are drawn i.i.d. from a "spike-and-slab" distribution.

    See:
        https://wesselb.github.io/assets/write-ups/Bruinsma,%20Spike%20and%20Slab%20Priors.pdf
    for a good review of the spike-and-slab model.

    Parameters
    ----------
    dim : int
        Number of weights per sample.
    p_spike : float
        The probability of the weight being 0.
    slab : Prior
        The distribution of the "slab".
        Since weights drawn from this distribution must be i.i.d.,
            we enforce `slab.D` to be 1.
    """

    def __init__(
        self,
        dim: int,
        p_spike: float,
        slab: Prior,
    ):
        if dim < 0:
            raise ValueError(f"`dim` should be nonnegative, got {dim}.")
        if p_spike < 0 or p_spike > 1:
            raise ValueError(f"Must have 0 <= `p_spike` <= 1, got `p_spike`={p_spike}.")
        if slab.D != 1:
            raise ValueError(
                f"`slab.D` must be 1 (got {slab.D}). "
                f"This enforces that can sample i.i.d. weights."
            )

        self.dim = dim
        self.p_spike = p_spike
        self.slab = slab

    @property
    def D(self):
        return self.dim

    def sample(self, num_samples: int):
        N = num_samples

        zero_weights = torch.zeros((N, self.D), dtype=torch.float32)
        slab_weights = self.slab.sample(num_samples * self.D)
        slab_weights = slab_weights.reshape((num_samples, self.D))

        spike_over_slab = torch.rand(N, self.D, dtype=torch.float32) < self.p_spike

        weights = torch.where(
            spike_over_slab,
            zero_weights,
            slab_weights,
        )

        return weights

    def log_prob(
        self,
        sample: torch.Tensor,
    ):
        super().check_sample_input(sample)

        N = sample.shape[0]

        log_prob = torch.zeros((N, self.D), dtype=torch.float32)

        spike_mask = sample == 0.
        slab_mask = sample != 0.

        # Add log-probability for spike.
        log_prob[spike_mask] = torch.log(torch.tensor(self.p_spike))

        # Add log-probability for slab.
        log_prob[slab_mask] = (
            self.slab.log_prob(sample[slab_mask].reshape(-1, 1)).reshape(-1)
            + torch.log(torch.tensor(1. - self.p_spike))
        )

        log_prob = torch.sum(log_prob, dim=1)  # [N]

        return log_prob
