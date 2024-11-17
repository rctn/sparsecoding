import torch
from typing import Dict


def compute_whitening_stats(X: torch.Tensor,
                            n_components=None):

    """
    Given a tensor of data, compute statistics for whitening transform.

    Args:
        X: Input data of size [N, D]
        n_components: Used for the PCA transform. Number of principal components to keep. If None, keep all components.
                        If int, keep that many components. If float between 0 and 1,
                        keep components that explain that fraction of variance.

    Returns:
        Dictionary containing whitening statistics
    """

    # Step 1: Center Data
    mean = torch.mean(X, dim=0)
    X_centered = X - mean
    Sigma = torch.cov(X_centered.T)

    # Step 2: Compute eigenvalues/eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

    # Since eigh returns values in ascending order, reverse them to get descending order
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvectors = torch.flip(eigenvectors, dims=[1])

    # Step 3: We provide the option of returning a certain
    # num of principal components. 0 <= n_components < 1 indicates you want to keep
    # a certain percentage of explained variance. n_components > 1 indicates a
    # you wish to keep that many. n_components = None means you want to keep all
    if n_components is not None:
        if isinstance(n_components, float):
            if not 0 < n_components <= 1:
                raise ValueError("If n_components is float, it must be between 0 and 1")
            explained_variance_ratio = eigenvalues / torch.sum(eigenvalues)
            cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
            n_components = torch.sum(cumulative_variance_ratio <= n_components) + 1
        elif isinstance(n_components, int):
            if not 0 < n_components <= len(eigenvalues):
                raise ValueError(f"n_components must be between 1 and {len(eigenvalues)}")
        else:
            raise ValueError("n_components must be int or float")

        # Instead of truncating, zero out unwanted components
        mask = torch.zeros_like(eigenvalues)
        mask[:n_components] = 1.0
        eigenvalues = eigenvalues * mask
        # For eigenvectors, we zero out the columns corresponding to zeroed eigenvalues
        eigenvectors = eigenvectors * mask.unsqueeze(0)

    return {
        'mean': mean,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
    }


def whiten(X: torch.Tensor,
           algorithm: str = 'zca',
           stats: Dict = None,
           n_components=None,
           epsilon: float = 0.
           ) -> torch.Tensor:

    """
    Apply whitening transform to data using pre-computed statistics.

    See https://arxiv.org/abs/1512.00809 for more details on transformations
    - Also gives two additional transforms that have not been implemented
    - Possible TODO

    See https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
    for details on PCA and ZCA in particular

    Args:
        X: Input data of shape [N, D] where N are unique data elements of dimensionality D 
        algorithm: Whitening transform we want to apply, one of ['zca', 'pca', or 'cholesky']
        stats: Dict containing precomputed whitening statistics (mean, eigenvectors, eigenvalues)
        n_components: number of components to retain if computing stats
        epsilon: Optional small constant to prevent division by zero

    Returns:
        Whitened data of shape [N, D] for ZCA and cholesky or [N, D_reduced] for PCA
        where D_reduced is the number of components kept
    """

    if stats is None:
        stats = compute_whitening_stats(X, n_components)

    x_centered = X - stats.get('mean')

    if algorithm == 'pca':
        # For PCA: project onto eigenvectors and scale
        scaling = torch.diag(1. / torch.sqrt(stats.get('eigenvalues') + epsilon))
        W = scaling @ stats.get('eigenvectors').T
    elif algorithm == 'zca':
        # For ZCA: project, scale, and rotate back
        scaling = torch.diag(1. / torch.sqrt(stats.get('eigenvalues') + epsilon))
        W = (stats.get('eigenvectors') @
             scaling @
             stats.get('eigenvectors').T)
    elif algorithm == 'cholesky':
        # Based on Cholesky decomp, also related to QR decomp
        scaling = torch.diag(1. / (stats.get('eigenvalues') + epsilon))
        W = torch.linalg.cholesky(stats.get('eigenvectors') @
                                  scaling @
                                  stats.get('eigenvectors').T).T
    else:
        raise ValueError(f"Unknown whitening algorithm: {algorithm}, must be one of ['pca', 'zca', 'cholesky]")

    return x_centered @ W.T
