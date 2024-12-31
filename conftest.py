import pytest
import torch

from sparsecoding.test_utils import (
    bars_datas_fixture,
    bars_datasets_fixture,
    bars_dictionary_fixture,
    dataset_size_fixture,
    patch_size_fixture,
    priors_fixture,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(1997)


# We import and define all fixtures in this file.
# This allows users to avoid any dependency fixtures.
# NOTE: This means pytest should only be run from this directory.
__all__ = [
    "bars_datas_fixture",
    "bars_datasets_fixture",
    "bars_dictionary_fixture",
    "dataset_size_fixture",
    "patch_size_fixture",
    "priors_fixture",
    "set_seed",
]
