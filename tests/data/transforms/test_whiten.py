import torch
import unittest

from data.transforms.whiten import Whitener


class TestWhitener(unittest.TestCase):
    def test_whitener(self):
        N = 100
        D = 10

        torch.manual_seed(1997)

        data = torch.rand((N, D), dtype=torch.float32)

        whitener = Whitener(data)

        whitened_data = whitener.whiten(data)
        assert torch.allclose(
            torch.mean(whitened_data, dim=0),
            torch.zeros(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have zero mean."
        assert torch.allclose(
            torch.cov(whitened_data.T),
            torch.eye(D, dtype=torch.float32),
            atol=1e-3,
        ), "Whitened data should have unit (identity) covariance."

        unwhitened_data = whitener.unwhiten(whitened_data)
        assert torch.allclose(
            unwhitened_data,
            data,
            atol=1e-3
        ), "Unwhitened data should be equal to input data."

    def test_zero_div(self):
        data = torch.Tensor([[1, 0], [2, 0]])
        whitener = Whitener(data)

        assert not torch.any(torch.isnan(
            whitener.whiten(data),
        )), "If an eigenvalue is 0, should not get NaNs."


if __name__ == "__main__":
    unittest.main()
