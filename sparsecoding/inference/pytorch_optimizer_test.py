import torch

from sparsecoding import inference
from sparsecoding.datasets import BarsDataset
from sparsecoding.test_utils import assert_allclose, assert_shape_equal


def lasso_loss(data, dictionary, coefficients, sparsity_penalty):
    """
    Generic MSE + l1-norm loss.
    """
    batch_size = data.shape[0]
    datahat = (dictionary @ coefficients.t()).t()

    mse_loss = torch.linalg.vector_norm(datahat - data, dim=1).square()
    sparse_loss = torch.sum(torch.abs(coefficients), axis=1)

    total_loss = (mse_loss + sparsity_penalty * sparse_loss) / batch_size
    return total_loss


def loss_fn(data, dictionary, coefficients):
    return lasso_loss(
        data,
        dictionary,
        coefficients,
        sparsity_penalty=1.0,
    )


def optimizer_fn(coefficients):
    return torch.optim.Adam(
        coefficients,
        lr=0.1,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )


def test_shape(
    patch_size_fixture: int,
    dataset_size_fixture: int,
    bars_dictionary_fixture: torch.Tensor,
    bars_datas_fixture: list[torch.Tensor],
    bars_datasets_fixture: list[BarsDataset],
):
    """
    Test that PyTorchOptimizer inference returns expected shapes.
    """
    for data, dataset in zip(bars_datas_fixture, bars_datasets_fixture):
        inference_method = inference.PyTorchOptimizer(
            optimizer_fn,
            loss_fn,
            n_iter=10,
        )
        a = inference_method.infer(data, bars_dictionary_fixture)
        assert_shape_equal(a, dataset.weights)


def test_inference(
    bars_dictionary_fixture: torch.Tensor,
    bars_datas_fixture: list[torch.Tensor],
    bars_datasets_fixture: list[BarsDataset],
):
    """
    Test that PyTorchOptimizer inference recovers the correct weights.
    """
    N_ITER = 1000

    for data, dataset in zip(bars_datas_fixture, bars_datasets_fixture):
        inference_method = inference.PyTorchOptimizer(
            optimizer_fn,
            loss_fn,
            n_iter=N_ITER,
        )

        a = inference_method.infer(data, bars_dictionary_fixture)

        assert_allclose(a, dataset.weights, atol=2e-1, rtol=1e-1)
