import torch
import unittest

from sparsecoding.priors import SpikeSlabPrior


class TestSpikeSlabPrior(unittest.TestCase):
    def test_spike_slab_prior(self):
        N = 10000
        D = 4
        p_spike = 0.5
        scale = 1.

        torch.manual_seed(1997)

        p_slab = 1. - p_spike

        for positive_only in [True, False]:
            spike_slab_prior = SpikeSlabPrior(
                D,
                p_spike,
                scale,
                positive_only,
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
            if positive_only:
                assert torch.sum(weights < 0.) == 0
            else:
                assert torch.allclose(
                    torch.sum(weights < 0.) / N_slab,
                    torch.sum(weights > 0.) / N_slab,
                    atol=2e-2,
                )
                weights = torch.abs(weights)

            laplace_weights = weights[weights > 0.]
            for quantile in torch.arange(5) / 5.:
                cutoff = -torch.log(1. - quantile)
                assert torch.allclose(
                    torch.sum(laplace_weights < cutoff) / N_slab,
                    quantile,
                    atol=1e-2,
                )


if __name__ == "__main__":
    unittest.main()
