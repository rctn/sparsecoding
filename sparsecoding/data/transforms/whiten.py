import torch


class Whitener(object):
    """Performs data whitening (and un-whitening).

    On initialization, stores the mean and covariance of the input data.
    These statistics are then used for future `whiten()` or `unwhiten()` calls.

    Parameters
    ----------
    data : Tensor, shape [N, D]
        Data to be whitened,
        where:
            N is the number of data points,
            D is the dimension of the data points.

    Attributes
    ----------
    mean : Tensor, shape [D]
        Mean of the input data for the Whitener.
    covariance : Tensor, shape [D, D]
        Co-variance of the input data for the Whitener.
    eigenvalues : Tensor, shape [D]
        Eigenvalues of `self.covariance`.
    eigenvectors : Tensor, shape [D, D]
        Eigenvectors of `self.covariance`.
    """

    def __init__(
        self,
        data,
    ):
        self.D = data.shape[1]

        with torch.no_grad():
            data = data.T  # [D, N]

            self.mean = torch.mean(data, dim=1)  # [D]

            self.covariance = torch.cov(data)  # [D, D]
            self.eigenvalues, self.eigenvectors = torch.linalg.eigh(self.covariance)  # [D], [D, D]

    def whiten(
        self,
        data: torch.Tensor,
    ):
        """Whitens the input `data` to have zero mean and unit (identity) covariance.

        Uses statistics of the data from class initialization.

        Note that this is a linear transformation; we use ZCA whitening.
        (
            See
            https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening  # noqa
            and
            https://jermwatt.github.io/control-notes/posts/zca_sphereing/ZCA_Sphereing.html  # noqa
            for good discussions on this vs. PCA whitening.
        ).

        Parameters
        ----------
        data : torch.Tensor, shape [N, D]
            Data to be whitened,
            where:
                N is the number of data points,
                D is the dimension of the data points.

        Returns
        -------
        whitened_data : Tensor, shape [N, D]
            Whitened data with zero mean and unit covariance.
        """
        if self.D != data.shape[1]:
            raise ValueError(
                f"`data` does not have same dimension as data given for class initialization"
                f"(got {data.shape[1]} and {self.D}, respectively)."
            )

        data = data.T  # [D, N]

        centered_data = data - self.mean.reshape(self.D, 1)  # [D, N]

        whitened_data = (
            self.eigenvectors
            @ torch.diag(1. / torch.sqrt(self.eigenvalues))
            @ self.eigenvectors.T
            @ centered_data
        )  # [D, N]

        return whitened_data.T

    def unwhiten(
        self,
        whitened_data: torch.Tensor,
    ):
        """
        Un-whitens the input `whitened_data`.

        Uses statistics of the data from class initialization.

        This is the inverse of `whiten()`.

        Parameters
        ----------
        whitened_data : Tensor, shape [N, D]
            Data to be un-whitened,
            where:
                N is the number of data points,
                D is the dimension of the data points.

        Returns
        -------
        unwhitened_data : Tensor, shape [N, D]
            Un-whitened data.
        """
        if self.D != whitened_data.shape[1]:
            raise ValueError(
                f"`whitened_data` does not have same dimension as data given for class initialization"
                f"(got {whitened_data.shape[1]} and {self.D}, respectively)."
            )

        whitened_data = whitened_data.T  # [D, N]

        unwhitened_data = (
            self.eigenvectors
            @ torch.diag(torch.sqrt(self.eigenvalues))
            @ self.eigenvectors.T
            @ whitened_data
        ) + self.mean.reshape(self.D, 1)  # [D, N]

        return unwhitened_data.T
