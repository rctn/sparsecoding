import torch

from sparsecoding.priors import L0Prior


def test_l0_prior():
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
