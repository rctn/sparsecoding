import pytest
import torch

from sparsecoding.priors import L0Prior, Prior, SpikeSlabPrior


@pytest.fixture()
def priors_fixture(patch_size_fixture: int) -> list[Prior]:
    return [
        SpikeSlabPrior(
            dim=2 * patch_size_fixture,
            p_spike=0.8,
            scale=1.0,
            positive_only=True,
        ),
        L0Prior(
            prob_distr=(
                torch.nn.functional.one_hot(
                    torch.tensor(1),
                    num_classes=2 * patch_size_fixture,
                ).type(torch.float32)
            ),
        ),
    ]