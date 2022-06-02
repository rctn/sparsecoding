from abc import ABC, abstractmethod

import torch


class Prior(ABC):
    """A distribution over weights.

    Parameters
    ----------
    weights_dim : int
        Number of weights for each sample.
    """
    @abstractmethod
    def D(self):
        """
        Number of weights per sample.
        """

    @abstractmethod
    def sample(
        self,
        num_samples: int = 1,
    ):
        """Sample weights from the prior.

        Parameters
        ----------
        num_samples : int, default=1
            Number of samples.

        Returns
        -------
        samples : Tensor, shape [num_samples, self.D()]
            Sampled weights.
        """

    @abstractmethod
    def log_prob(
        self,
        sample: torch.Tensor,
    ):
        """Get the log-probability of the sample under this distribution.

        Parameters
        ----------
        sample : Tensor, shape [num_samples, self.D()]
            Sample to get the log-probability for.

        Returns
        -------
        log_prob : Tensor, shape [num_samples]
            Log-probability of `sample`.
        """

    def check_sample_input(
        self,
        sample: torch.Tensor,
    ):
        """Check the shape and dtype of the sample.

        Used in:
            self.log_prob().
        """
        if sample.dtype != torch.float32:
            raise ValueError(f"`sample` dtype should be float32, got {sample.dtype}.")
        if sample.dim() != 2 or sample.shape[1] != self.D:
            raise ValueError(f"`sample` should have shape [N, {self.D}], got {sample.shape}.")
