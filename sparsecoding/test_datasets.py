import pytest
from sparsecoding.datasets import FieldDataset


@pytest.mark.parametrize("whitened", [True, False])
def test_FieldDataset(
        patch_size_fixture: int,
        dataset_size_fixture: int,
        whitened: bool,
):
    fielddataset = FieldDataset(
        root="data",
        num_patches=dataset_size_fixture,
        patch_size=patch_size_fixture,
        whitened=whitened
    )
    assert len(fielddataset) == dataset_size_fixture
    assert fielddataset.patches.shape == (dataset_size_fixture, 1, patch_size_fixture, patch_size_fixture)
