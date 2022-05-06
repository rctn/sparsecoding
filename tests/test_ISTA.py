from tests.testing_utilities import TestCase
from tests.data_generation import BarsDataset
import torch
from sparsecoding import inference


class TestISTA(TestCase):
    '''Test ISTA inference algorithm'''

    def test_coefficient_shapes(self):

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertShapeEqual(a, bars.coefficients)

        inference_method = inference.ISTA(n_iter=10)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))

        inference_method = inference.ISTA(n_iter=10, stop_early=True)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))

    def test_bars(self):
        '''Evaluate quality of coefficient inference on bars dataset'''
        inference_method = inference.ISTA(coeff_lr=5e-3, n_iter=100)
        cpudevice = torch.device('cpu')
        # coefficients do not go identically to zero - very relaxed criteria
        rtol = 1e-4
        atol = 1e-4

        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data, bars.dictionary)
            self.assertAllClose(a.to(cpudevice), bars.coefficients.to(
                cpudevice), rtol=rtol, atol=atol)

        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
