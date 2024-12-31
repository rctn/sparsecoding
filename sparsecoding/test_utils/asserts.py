import numpy as np

# constants
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5

def assert_allclose(a: np.ndarray, b: np.ndarray, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> None:
    return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def assert_shape_equal(a: np.ndarray, b: np.ndarray) -> None:
    assert a.shape == b.shape