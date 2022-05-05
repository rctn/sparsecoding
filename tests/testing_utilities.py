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
        