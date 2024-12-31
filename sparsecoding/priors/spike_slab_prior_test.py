import pytest
import torch

from sparsecoding.priors import SpikeSlabPrior


@pytest.mark.parametrize("positive_only", [True, False])
def test_spike_slab_prior(positive_only: bool):
    N = 10000
    D = 4
    p_spike = 0.5
    scale = 1.0

    p_slab = 1.0 - p_spike

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
        torch.sum(weights == 0.0) / (N * D),
        torch.tensor(p_spike),
        atol=1e-2,
    )

    # Check Laplacian distribution.
    N_slab = p_slab * N * D
    if positive_only:
        assert torch.sum(weights < 0.0) == 0
    else:
        assert torch.allclose(
            torch.sum(weights < 0.0) / N_slab,
            torch.sum(weights > 0.0) / N_slab,
            atol=2e-2,
        )
        weights = torch.abs(weights)

    laplace_weights = weights[weights > 0.0]
    for quantile in torch.arange(5) / 5.0:
        cutoff = -torch.log(1.0 - quantile)
        assert torch.allclose(
            torch.sum(laplace_weights < cutoff) / N_slab,
            quantile,
            atol=1e-2,
        )
