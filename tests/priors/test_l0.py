from itertools import product

import torch
import unittest

from sparsecoding.priors.l0 import L0Prior


class TestL0Prior(unittest.TestCase):
    def test_sample(self):
        N = 10000
        prob_distr = torch.tensor([0.5, 0.25, 0, 0.25])

        torch.manual_seed(1997)

        D = prob_distr.shape[0]

        l0_prior = L0Prior(prob_distr)
        weights = l0_prior.sample(N)

        assert weights.shape == (N, D)

        # Check uniform distribution over which weights are active.
        per_weight_hist = torch.sum(weights, dim=0)  # [D]
        normalized_per_weight_hist = per_weight_hist / torch.sum(per_weight_hist)  # [D]
        assert torch.allclose(
            normalized_per_weight_hist,
            torch.full((D,), 0.25, dtype=torch.float32),
            atol=1e-2,
        )

        # Check the distribution over the l0-norm of the weights.
        num_active_per_sample = torch.sum(weights, dim=1)  # [N]
        for num_active in range(1, 5):
            assert torch.allclose(
                torch.sum(num_active_per_sample == num_active) / N,
                prob_distr[num_active - 1],
                atol=1e-2,
            )

    def test_log_prob(self):
        prob_distr = torch.tensor([0.75, 0.25, 0.])

        l0_prior = L0Prior(prob_distr)

        samples = list(product([0, 1], repeat=3))  # [2**D, D]
        samples = torch.tensor(samples, dtype=torch.float32)  # [2**D, D]

        log_probs = l0_prior.log_prob(samples)

        # The l0-norm at index `i`
        # is the number of ones
        # in the binary representation of `i`.
        assert log_probs[0] == -torch.inf
        assert torch.allclose(
            log_probs[[1, 2, 4]],
            torch.log(torch.tensor(0.75)),
        )
        assert torch.allclose(
            log_probs[[3, 5, 6]],
            torch.log(torch.tensor(0.25)),
        )
        assert log_probs[7] == -torch.inf


if __name__ == "__main__":
    unittest.main()
