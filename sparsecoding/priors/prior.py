from abc import ABC, abstractmethod


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
        samples : Tensor, shape [num_samples, self.D]
            Sampled weights.
        """
