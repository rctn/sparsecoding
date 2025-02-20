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
    Test that Vanilla inference returns expected shapes.
    """
    N_ITER = 10

    for (data, dataset) in zip(bars_datas_fixture, bars_datasets_fixture):
        inference_method = inference.Vanilla(N_ITER)
        a = inference_method.infer(data, bars_dictionary_fixture)
        assert_shape_equal(a, dataset.weights)

        inference_method = inference.Vanilla(N_ITER, return_all_coefficients=True)
        a = inference_method.infer(data, bars_dictionary_fixture)
        assert a.shape == (dataset_size_fixture, N_ITER + 1, 2 * patch_size_fixture)


def test_inference(
    bars_dictionary_fixture: torch.Tensor,
    bars_datas_fixture: list[torch.Tensor],
    bars_datasets_fixture: list[BarsDataset],
):
    """
    Test that Vanilla inference recovers the correct weights.
    """
    LR = 5e-2
    N_ITER = 1000

    for (data, dataset) in zip(bars_datas_fixture, bars_datasets_fixture):
        inference_method = inference.Vanilla(coeff_lr=LR, n_iter=N_ITER)

        a = inference_method.infer(data, bars_dictionary_fixture)

        assert_allclose(a, dataset.weights, atol=5e-2)
