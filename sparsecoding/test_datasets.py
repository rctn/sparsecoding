from sparsecoding.datasets import FieldDataset


def test_FieldDataset(
        patch_size_fixture: int,
        dataset_size_fixture: int,
):
    fielddataset = FieldDataset(
        root="data/",
        num_patches=dataset_size_fixture,
        patch_size=patch_size_fixture,
    )
    assert len(fielddataset) == dataset_size_fixture
    assert fielddataset.patches.shape == (dataset_size_fixture, 1, patch_size_fixture, patch_size_fixture)
