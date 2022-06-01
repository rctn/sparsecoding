import torch
import unittest

from sparsecoding.priors.laplace import LaplacePrior


class TestLaplacePrior(unittest.TestCase):
    def test_sample(self):
        N = 10000
        D = 4
        scale = 1.

        torch.manual_seed(1997)

        for positive_only in [True, False]:
            laplace_prior = LaplacePrior(
                D,
                scale,
                positive_only,
            )
            weights = laplace_prior.sample(N)

            assert weights.shape == (N, D)

            # Check Laplacian distribution.
            if positive_only:
                assert torch.sum(weights < 0.) == 0
            else:
                assert torch.allclose(
                    torch.sum(weights < 0.) / (N * D),
                    torch.sum(weights > 0.) / (N * D),
                    atol=2e-2,
                )
                weights = torch.abs(weights)

            laplace_weights = weights[weights > 0.]
            for quantile in torch.arange(5) / 5.:
                cutoff = -torch.log(1. - quantile)
                assert torch.allclose(
                    torch.sum(laplace_weights < cutoff) / (N * D),
                    quantile,
                    atol=1e-2,
                )

    def test_log_prob(self):
        D = 3
        scale = 1.

        samples = torch.Tensor([[-1., 0., 1.]])

        pos_only_log_prob = torch.tensor(-2.)

        for positive_only in [True, False]:
            laplace_prior = LaplacePrior(
                D,
                scale,
                positive_only,
            )

            if positive_only:
                assert laplace_prior.log_prob(samples)[0] == -torch.inf

                samples = torch.abs(samples)
                assert torch.allclose(
                    laplace_prior.log_prob(samples)[0],
                    pos_only_log_prob,
                )
            else:
                assert torch.allclose(
                    laplace_prior.log_prob(samples)[0],
                    pos_only_log_prob - D * torch.log(torch.tensor(2.)),
                )


if __name__ == "__main__":
    unittest.main()
