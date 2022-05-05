from tests.testing_utilities import TestCase,CommonInferenceTests
import torch
import numpy as np

from sparsecoding import inference


class TestLCA(TestCase,CommonInferenceTests):
    
    def test_bars(self):
        lca = inference.LCA(n_iter=100,coeff_lr=1e-1)
        self.assertBarInference(inference_method=lca)
        