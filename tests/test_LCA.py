from tests.testing_utilities import TestCase
from tests.data_generation import BarsDataset

import torch
import numpy as np

from sparsecoding import inference


class TestLCA(TestCase):
    '''Test Locally Competative Algorithm'''

    
    def test_coefficient_shapes(self):
        
        def evaluate(device):
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data,bars.dictionary)
            self.assertShapeEqual(a,bars.coefficients)

        inference_method = inference.LCA()
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
        
        inference_method = inference.LCA(n_iter=10,stop_early=True)
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
            
    
    def test_bars(self):
        '''Evaluate quality of coefficient inference on bars dataset'''
        inference_method = inference.LCA(coeff_lr=1e-2,threshold=0.1,n_iter=100)
        cpudevice = torch.device('cpu')
        rtol = 1e-2
        atol = 1e-2

        def evaluate(device):    
            bars = BarsDataset(device=device)
            a = inference_method.infer(bars.data,bars.dictionary)
            self.assertAllClose(a.to(cpudevice),bars.coefficients.to(cpudevice),rtol=rtol,atol=atol)
        
        evaluate(torch.device('cpu'))
        if torch.cuda.is_available():
            evaluate(torch.device('cuda'))
