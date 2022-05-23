from tests.testing_utilities import TestCase
from tests.data_generation import BarsDataset

import torch

from sparsecoding import inference


class TestEuler(TestCase):
    '''Test Euler inference algorithm'''

    def test_coefficient_shapes(self):

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertShapeEqual(a, bars.coefficients)

        def evalute_return_all_coefficients(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertEqual(a.shape, (bars.n_samples, inference_method.n_iter+1, bars.n_basis))

        # generic
        inference_method = inference.Euler(n_iter=10)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))

        # stop early condition
        inference_method = inference.Euler(n_iter=10, stop_early=True)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))

        # return_all_coefficients=True
        inference_method = inference.Euler(n_iter=10, return_all_coefficients=True)
        evalute_return_all_coefficients(torch.device('cpu'))
        if torch.cuda.is_available():
            evalute_return_all_coefficients(torch.device('cuda'))

    def test_bars(self):
        '''Evaluate quality of coefficient inference on bars dataset'''
        inference_method = inference.Euler(coeff_lr=5e-3, n_iter=100)
        cpudevice = torch.device('cpu')
        # coefficients do not go identically to zero - very relaxed criteria
        rtol = 1e-1
        atol = 1e-1

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertAllClose(a.to(cpudevice), bars.coefficients.to(cpudevice), rtol=rtol, atol=atol)

        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
