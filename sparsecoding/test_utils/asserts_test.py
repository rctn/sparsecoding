
import numpy as np
import torch

from .asserts import assert_allclose, assert_shape_equal


def test_pytorch_all_close():
    result = torch.ones([10, 10]) + 1e-10
    expected = torch.ones([10, 10])
    assert_allclose(result, expected)

def test_np_all_close():
    result = np.ones([100, 100]) + 1e-10
    expected = np.ones([100, 100])
    assert_allclose(result, expected)

def test_assert_pytorch_shape_equal():
    a = torch.zeros([10, 10])
    b = torch.ones([10, 10])
    assert_shape_equal(a, b)

def test_assert_np_shape_equal():
    a = np.zeros([100, 100])
    b = np.ones([100, 100])
    assert_shape_equal(a, b)
