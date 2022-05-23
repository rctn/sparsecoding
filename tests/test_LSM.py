from tests.testing_utilities import TestCase
from tests.data_generation import BarsDataset

import torch

from sparsecoding import inference


class TestLSM(TestCase):
    '''Test Locally Competative Algorithm'''

    def test_coefficient_shapes(self):

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertShapeEqual(a, bars.coefficients)

        inference_method = inference.LSM(n_iter=10)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))

        inference_method = inference.LSM(n_iter=10)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))

    def test_bars(self):
        '''Evaluate quality of coefficient inference on bars dataset'''
        inference_method = inference.LSM(n_iter=100)
        cpudevice = torch.device('cpu')
        rtol = 1e-0
        atol = 1e-0

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertAllClose(a.to(cpudevice), bars.coefficients.to(cpudevice), rtol=rtol, atol=atol)

        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
