import torch

from sparsecoding import inference
from sparsecoding.datasets import BarsDataset
from sparsecoding.test_utils import assert_allclose, assert_shape_equal


def test_shape(
    patch_size_fixture: int,
    dataset_size_fixture: int,
    bars_dictionary_fixture: torch.Tensor,
    bars_datas_fixture: list[torch.Tensor],
    bars_datasets_fixture: list[BarsDataset],
):
    """
    Test that LSM inference returns expected shapes.
    """
    N_ITER = 10

    for data, dataset in zip(bars_datas_fixture, bars_datasets_fixture):
        inference_method = inference.LSM(N_ITER)
        a = inference_method.infer(data, bars_dictionary_fixture)
        assert_shape_equal(a, dataset.weights)


def test_inference(
    bars_dictionary_fixture: torch.Tensor,
    bars_datas_fixture: list[torch.Tensor],
    bars_datasets_fixture: list[BarsDataset],
):
    """
    Test that LSM inference recovers the correct weights.
    """
    N_ITER = 1000

    for data, dataset in zip(bars_datas_fixture, bars_datasets_fixture):
        inference_method = inference.LSM(n_iter=N_ITER)

        a = inference_method.infer(data, bars_dictionary_fixture)

        assert_allclose(a, dataset.weights, atol=6e-2)
