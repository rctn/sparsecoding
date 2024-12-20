import torch
import unittest

from sparsecoding.transforms import whiten


class TestWhitener(unittest.TestCase):
    def test_zca(self):
        N = 5000
        D = 32*32

        torch.manual_seed(1997)

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

    def test_pca(self):
        N = 5000
        D = 32*32

        torch.manual_seed(1997)

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

    def test_cholesky(self):
        N = 5000
        D = 32*32

        torch.manual_seed(1997)

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


if __name__ == "__main__":
    unittest.main()
