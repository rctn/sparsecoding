import numpy as np
import torch
import unittest
from tests.data_generation import BarsDataset


## constants
default_atol = 1e-6
default_rtol = 1e-5


class TestCase(unittest.TestCase):
    '''Base class for testing'''

    def assertAllClose(self, a, b, rtol=default_rtol, atol=default_atol):
        return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

    def assertShapeEqual(self, a, b):
        assert a.shape == b.shape
        
        
        
class CommonInferenceTests(object):
    '''Base class for evaluating inference method'''
    
    def assertBarInference(self,inference_method,patch_size=8,n_samples=2,coefficient_treshold=0.8,device=torch.device('cpu')):
        cpudevice = torch.device('cpu')

        bars = BarsDataset(patch_size=patch_size,
                           n_samples=n_samples,
                           device=device,
                           coefficient_treshold=coefficient_treshold
        )
        a = inference_method.infer(bars.data,bars.dictionary)

        self.assertAllClose(a.to(cpudevice),bars.coefficients.to(cpudevice))
    