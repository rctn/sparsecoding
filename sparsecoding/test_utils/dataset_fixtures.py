
import pytest
import torch

from sparsecoding.datasets import BarsDataset
from sparsecoding.priors import Prior


@pytest.fixture()
def bars_datasets_fixture(patch_size_fixture: int, dataset_size_fixture: int, priors_fixture: list[Prior]) -> list[BarsDataset]:
    return [
        BarsDataset(
            patch_size=patch_size_fixture,
            dataset_size=dataset_size_fixture,
            prior=prior,
        )
        for prior in priors_fixture
    ]

@pytest.fixture()
def bars_datas_fixture(patch_size_fixture: int, dataset_size_fixture: int, bars_datasets_fixture: list[BarsDataset]) -> list[torch.Tensor]:
    return [
        dataset.data.reshape((dataset_size_fixture, patch_size_fixture * patch_size_fixture))
        for dataset in bars_datasets_fixture
    ]