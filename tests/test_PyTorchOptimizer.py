from tests.testing_utilities import TestCase
from tests.data_generation import BarsDataset

import torch

from sparsecoding import inference


class TestPyTorchOptimizer(TestCase):
    '''Test PyTorchOptimizer'''

    def generic_sc_loss(self, data, dictionary, coefficients, sparsity_penalty):
        '''NOT A TEST - Generic MSE and l-1 loss function'''
        batch_size = data.shape[0]
        datahat = (dictionary@coefficients.t()).t()

        mse_loss = torch.linalg.vector_norm(datahat-data, dim=1).square()
        sparse_loss = torch.sum(torch.abs(coefficients), axis=1)

        total_loss = (mse_loss + sparsity_penalty*sparse_loss)/batch_size
        return total_loss

    def test_coefficient_shapes(self):
        # define the loss function and optimizer with chosen parameters
        def loss_f(data, dictionary, coefficients): return self.generic_sc_loss(
            data, dictionary, coefficients, sparsity_penalty=1.)

        def optimizer_f(coefficients): return torch.optim.Adam(
            coefficients, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertShapeEqual(a, bars.coefficients)

        inference_method = inference.PyTorchOptimizer(optimizer_f, loss_f, n_iter=10)
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
        else:
            evaluate(torch.device('cpu'))

    def test_bars(self):
        '''Evaluate quality of coefficient inference on bars dataset'''
        def loss_f(data, dictionary, coefficients): return self.generic_sc_loss(
            data, dictionary, coefficients, sparsity_penalty=1.)

        def optimizer_f(coefficients): return torch.optim.Adam(
            coefficients, lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        inference_method = inference.PyTorchOptimizer(optimizer_f, loss_f, n_iter=100)
        cpudevice = torch.device('cpu')
        rtol = 1e-1
        atol = 1e-1

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertAllClose(a.to(cpudevice), bars.coefficients.to(cpudevice), rtol=rtol, atol=atol)

        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
