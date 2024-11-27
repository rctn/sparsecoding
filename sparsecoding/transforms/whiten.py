import torch
from typing import Dict


def compute_whitening_stats(X: torch.Tensor):

    """
    Given a tensor of data, compute statistics for whitening transform.

    Parameters
    ----------
    X: Input data of size [N, D]

    Returns
    ----------
    Dictionary containing whitening statistics (eigenvalues, eigenvectors, mean)
    """

    mean = torch.mean(X, dim=0)
    X_centered = X - mean
    Sigma = torch.cov(X_centered.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

    # Since eigh returns values in ascending order, reverse them to get descending order
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])

    return {
        'mean': mean,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'covariance': Sigma
    }


def whiten(X: torch.Tensor,
           algorithm: str = 'zca',
           stats: Dict = None,
           n_components=None,
           epsilon: float = 0.
           ) -> torch.Tensor:

    """
    Apply whitening transform to data using pre-computed statistics.

    Parameters
    ----------
    X: Input data of shape [N, D] where N are unique data elements of dimensionality D
    algorithm: Whitening transform we want to apply, one of ['zca', 'pca', or 'cholesky']
    stats: Dict containing precomputed whitening statistics (mean, eigenvectors, eigenvalues)
    n_components: Number of principal components to keep. If None, keep all components.
                  If int, keep that many components. If float between 0 and 1,
                  keep components that explain that fraction of variance.
    epsilon: Optional small constant to prevent division by zero

    Returns
    ----------
    Whitened data of shape [N, D]

    Notes
    ----------
    See https://arxiv.org/abs/1512.00809 for more details on transformations
    - Also gives two additional transforms that have not been implemented
    - Possible TODO

    See https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
    for details on PCA and ZCA in particular
    """

    if stats is None:
        stats = compute_whitening_stats(X)

    x_centered = X - stats.get('mean')

    if algorithm == 'pca' or algorithm == 'zca':

        scaling = 1. / torch.sqrt(stats.get('eigenvalues') + epsilon)

        if n_components is not None:
            if isinstance(n_components, float):
                if not 0 < n_components <= 1:
                    raise ValueError("If n_components is float, it must be between 0 and 1")
                explained_variance_ratio = stats.get('eigenvalues') / torch.sum(stats.get('eigenvalues'))
                cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
                n_components = torch.sum(cumulative_variance_ratio <= n_components) + 1
            elif isinstance(n_components, int):
                if not 0 < n_components <= len(stats.get('eigenvalues')):
                    raise ValueError(f"n_components must be between 1 and {len(stats.get('eigenvalues'))}")
            else:
                raise ValueError("n_components must be int or float")

            mask = torch.zeros_like(scaling)
            mask[:n_components] = 1.0
            scaling = scaling * mask

        scaling = torch.diag(scaling)

        if algorithm == 'pca':
            # For PCA: project onto eigenvectors and scale
            W = scaling @ stats.get('eigenvectors').T
        else:
            # For ZCA: project, scale, and rotate back
            W = (stats.get('eigenvectors') @
                 scaling @
                 stats.get('eigenvectors').T)
    elif algorithm == 'cholesky':
        # Based on Cholesky decomp, also related to QR decomp
        L = torch.linalg.cholesky(stats.get('covariance'))
        W = torch.linalg.inv(L)
    else:
        raise ValueError(f"Unknown whitening algorithm: {algorithm}, must be one of ['pca', 'zca', 'cholesky]")

    return x_centered @ W.T
