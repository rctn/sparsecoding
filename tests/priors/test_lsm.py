import numpy as np
import torch
import unittest

from sparsecoding.priors.lsm import LSMPrior


class TestLSMPrior(unittest.TestCase):
    def test_sample(self):
        N = 10000
        D = 4
        alpha = 2
        beta = 2

        torch.manual_seed(1997)

        for positive_only in [True, False]:
            lsm_prior = LSMPrior(
                D,
                alpha,
                beta,
                positive_only,
            )
            weights = lsm_prior.sample(N)

            assert weights.shape == (N, D)

            # Check distribution.
            if positive_only:
                assert torch.sum(weights < 0.) == 0
            else:
                assert torch.allclose(
                    torch.sum(weights < 0.) / (N * D),
                    torch.sum(weights > 0.) / (N * D),
                    atol=2e-2,
                )
                weights = torch.abs(weights)

            # Note:
            #     Antiderivative of positive-only is:
            #         -Beta^alpha * (Beta + x)^(-alpha),
            #     cdf is:
            #         1. - Beta^alpha * (B + x)^(-alpha),
            #     quantile fn is:
            #        -Beta + exp((log(1-y) - alpha*log(Beta)) / -alpha)

            for quantile in torch.arange(5) / 5.:
                cutoff = (
                    -beta
                    + np.exp(
                        (np.log(1. - quantile) - alpha * np.log(beta))
                        / (-alpha)
                    )
                )
                assert torch.allclose(
                    torch.sum(weights < cutoff) / (N * D),
                    quantile,
                    atol=1e-2,
                )

    def test_log_prob(self):
        D = 3
        alpha = 2
        beta = 2

        samples = torch.Tensor([[-1., 0., 1.]])

        pos_only_log_prob = (
            torch.log(torch.tensor(alpha)) - torch.log(torch.tensor(beta))
            + 2 * (
                torch.log(torch.tensor(alpha)) + alpha * torch.log(torch.tensor(beta))
                - (alpha + 1) * torch.log(torch.tensor(1 + beta))
            )
        )

        for positive_only in [True, False]:
            lsm_prior = LSMPrior(
                D,
                alpha,
                beta,
                positive_only,
            )

            if positive_only:
                assert lsm_prior.log_prob(samples)[0] == -torch.inf

                samples = torch.abs(samples)
                assert torch.allclose(
                    lsm_prior.log_prob(samples)[0],
                    pos_only_log_prob,
                )
            else:
                assert torch.allclose(
                    lsm_prior.log_prob(samples)[0],
                    pos_only_log_prob - D * torch.log(torch.tensor(2.)),
                )


if __name__ == "__main__":
    unittest.main()
