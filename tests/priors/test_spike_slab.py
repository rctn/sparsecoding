from operator import pos
import torch
import unittest

from sparsecoding.priors.laplace import LaplacePrior
from sparsecoding.priors.spike_slab import SpikeSlabPrior


class TestSpikeSlabPrior(unittest.TestCase):
    def test_sample(self):
        N = 10000
        D = 4
        p_spike = 0.5
        slab = LaplacePrior(
            dim=1,
            scale=1.0,
            positive_only=True,
        )

        torch.manual_seed(1997)

        p_slab = 1. - p_spike

        spike_slab_prior = SpikeSlabPrior(
            D,
            p_spike,
            slab,
        )
        weights = spike_slab_prior.sample(N)

        assert weights.shape == (N, D)

        # Check spike probability.
        assert torch.allclose(
            torch.sum(weights == 0.) / (N * D),
            torch.tensor(p_spike),
            atol=1e-2,
        )

        # Check Laplacian distribution.
        N_slab = p_slab * N * D
        assert torch.sum(weights < 0.) == 0

        laplace_weights = weights[weights > 0.]
        for quantile in torch.arange(5) / 5.:
            cutoff = -torch.log(1. - quantile)
            assert torch.allclose(
                torch.sum(laplace_weights < cutoff) / N_slab,
                quantile,
                atol=1e-2,
            )

    def test_log_prob(self):
        D = 3
        p_spike = 0.5
        slab = LaplacePrior(
            dim=1,
            scale=1.0,
            positive_only=True,
        )

        samples = torch.Tensor([[-1., 0., 1.]])

        spike_slab_prior = SpikeSlabPrior(
            D,
            p_spike,
            slab,
        )

        assert spike_slab_prior.log_prob(samples)[0] == -torch.inf

        samples = torch.abs(samples)
        assert torch.allclose(
            spike_slab_prior.log_prob(samples)[0],
            (
                torch.log(torch.tensor(p_spike))
                + 2 * (
                    -1. + torch.log(torch.tensor(1. - p_spike))
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
