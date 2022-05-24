import torch
import unittest

from sparsecoding import inference
from tests.testing_utilities import TestCase
from tests.inference.common import DATAS, DATASET, DICTIONARY


class TestPyTorchOptimizer(TestCase):
    def lasso_loss(data, dictionary, coefficients, sparsity_penalty):
        """
        Generic MSE + l1-norm loss.
        """
        batch_size = data.shape[0]
        datahat = (dictionary@coefficients.t()).t()

        mse_loss = torch.linalg.vector_norm(datahat-data, dim=1).square()
        sparse_loss = torch.sum(torch.abs(coefficients), axis=1)

        total_loss = (mse_loss + sparsity_penalty*sparse_loss)/batch_size
        return total_loss

    def loss_fn(data, dictionary, coefficients):
        return TestPyTorchOptimizer.lasso_loss(
            data,
            dictionary,
            coefficients,
            sparsity_penalty=1.,
        )

    def optimizer_fn(coefficients):
        return torch.optim.Adam(
            coefficients,
            lr=0.1,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

    def test_shape(self):
        """
        Test that PyTorchOptimizer inference returns expected shapes.
        """
        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.PyTorchOptimizer(
                TestPyTorchOptimizer.optimizer_fn,
                TestPyTorchOptimizer.loss_fn,
                n_iter=10,
            )
            a = inference_method.infer(data, DICTIONARY)
            self.assertShapeEqual(a, dataset.weights)

    def test_inference(self):
        """
        Test that PyTorchOptimizer inference recovers the correct weights.
        """
        N_ITER = 1000

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.PyTorchOptimizer(
                TestPyTorchOptimizer.optimizer_fn,
                TestPyTorchOptimizer.loss_fn,
                n_iter=N_ITER,
            )

            a = inference_method.infer(data, DICTIONARY)

            self.assertAllClose(a, dataset.weights, atol=1e-1, rtol=1e-1)

if __name__ == "__main__":
    unittest.main()
