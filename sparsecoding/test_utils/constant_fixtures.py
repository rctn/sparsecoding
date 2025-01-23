import pytest

PATCH_SIZE = 8
DATASET_SIZE = 1000


@pytest.fixture()
def patch_size_fixture() -> int:
    return PATCH_SIZE


@pytest.fixture()
def dataset_size_fixture() -> int:
    return DATASET_SIZE
