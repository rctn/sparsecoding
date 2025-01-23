import pytest
import torch

from sparsecoding.datasets import BarsDataset


@pytest.fixture()
def bars_dictionary_fixture(patch_size_fixture: int, bars_datasets_fixture: list[BarsDataset]) -> torch.Tensor:
    """Return a bars dataset basis reshaped to represent a dictionary."""
    return bars_datasets_fixture[0].basis.reshape((2 * patch_size_fixture, patch_size_fixture * patch_size_fixture)).T
