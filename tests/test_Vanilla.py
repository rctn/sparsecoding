from tests.testing_utilities import TestCase,CommonInferenceTests
import torch
import numpy as np

from sparsecoding import inference


class TestVanilla(TestCase,CommonInferenceTests):
    
    def test_bars(self):
        vanilla = inference.Vanilla(coeff_lr=1e-2,sparsity_penalty=1,n_iter=300)
        self.assertBarInference(inference_method=vanilla)
        