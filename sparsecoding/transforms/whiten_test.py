import torch

from sparsecoding.transforms import whiten


def test_zca():
    N = 5000
    D = 32*32

    X = torch.rand((N, D), dtype=torch.float32)

    X_whitened = whiten(X)

    assert torch.allclose(
        torch.mean(X_whitened, dim=0),
        torch.zeros(D, dtype=torch.float32),
        atol=1e-3,
    ), "Whitened data should have zero mean."
    assert torch.allclose(
        torch.cov(X_whitened.T),
        torch.eye(D, dtype=torch.float32),
        atol=1e-3,
    ), "Whitened data should have unit (identity) covariance."

def test_pca():
    N = 5000
    D = 32*32

    X = torch.rand((N, D), dtype=torch.float32)

    X_whitened = whiten(X, algorithm='pca')

    assert torch.allclose(
        torch.mean(X_whitened, dim=0),
        torch.zeros(D, dtype=torch.float32),
        atol=1e-3,
    ), "Whitened data should have zero mean."
    assert torch.allclose(
        torch.cov(X_whitened.T),
        torch.eye(D, dtype=torch.float32),
        atol=1e-3,
    ), "Whitened data should have unit (identity) covariance."

def test_cholesky():
    N = 5000
    D = 32*32

    X = torch.rand((N, D), dtype=torch.float32)

    X_whitened = whiten(X, algorithm='cholesky')

    assert torch.allclose(
        torch.mean(X_whitened, dim=0),
        torch.zeros(D, dtype=torch.float32),
        atol=1e-3,
    ), "Whitened data should have zero mean."
    assert torch.allclose(
        torch.cov(X_whitened.T),
        torch.eye(D, dtype=torch.float32),
        atol=1e-3,
    ), "Whitened data should have unit (identity) covariance."
