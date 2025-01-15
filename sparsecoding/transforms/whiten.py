from typing import Literal, overload

import torch


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

    return {"mean": mean, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "covariance": Sigma}


@overload
def whiten(
    X: torch.Tensor,
    algorithm: str = "zca",
    stats: dict | None = None,
    n_components: float | None = None,
    epsilon: float = 0.0,
    return_W: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def whiten(
    X: torch.Tensor,
    algorithm: str = "zca",
    stats: dict | None = None,
    n_components: float | None = None,
    epsilon: float = 0.0,
    return_W: Literal[True] = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...


# The last overload is a fallback in case the caller
# provides a regular bool:
@overload
def whiten(
    X: torch.Tensor,
    algorithm: str = "zca",
    stats: dict | None = None,
    n_components: float | None = None,
    epsilon: float = 0.0,
    return_W: bool = False,
) -> torch.Tensor: ...


def whiten(
    X: torch.Tensor,
    algorithm: str = "zca",
    stats: dict | None = None,
    n_components: float | None = None,
    epsilon: float = 0.0,
    return_W: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Apply whitening transform to data using pre-computed statistics.

    Parameters
    ----------
    X: Input data of shape [N, D] where N are unique data elements of dimensionality D
    algorithm: Whitening transform we want to apply, one of ['zca', 'pca', or 'cholesky']
    stats: dict containing precomputed whitening statistics (mean, eigenvectors, eigenvalues)
    n_components: Number of principal components to keep. If None, keep all components.
                  If int, keep that many components. If float between 0 and 1,
                  keep components that explain that fraction of variance.
    epsilon: Optional small constant to prevent division by zero

    Returns
    ----------
    Whitened data of shape [N, D]

    Notes
    ----------
    See examples/Data_Whitening.ipynb for usage examples, and brief discussion about the different whitening methods

    See https://arxiv.org/abs/1512.00809 for extensive details on whitening transformations
    - Possible TODO: Also gives two additional transforms that have not been implemented

    See https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening
    for details on PCA and ZCA in particular
    """

    if stats is None:
        stats = compute_whitening_stats(X)

    x_centered = X - stats.get("mean")

    if algorithm == "pca" or algorithm == "zca":

        eigenvalues = stats.get("eigenvalues", None)
        assert eigenvalues is not None  # type narrowing
        scaling = 1.0 / torch.sqrt(eigenvalues + epsilon)

        if n_components is not None:
            if isinstance(n_components, float):
                if not 0 < n_components <= 1:
                    raise ValueError("If n_components is float, it must be between 0 and 1")
                explained_variance_ratio = stats.get("eigenvalues") / torch.sum(eigenvalues)
                cumulative_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)
                n_components = float(torch.sum(cumulative_variance_ratio <= n_components)) + 1
            elif isinstance(n_components, int):
                if not 0 < n_components <= len(eigenvalues):
                    raise ValueError(f"n_components must be between 1 and {len(eigenvalues)}")
            else:
                raise ValueError("n_components must be int or float")

            mask = torch.zeros_like(scaling)
            mask[:n_components] = 1.0
            scaling = scaling * mask

        scaling = torch.diag(scaling)
        eigenvectors = stats.get("eigenvectors", None)
        assert eigenvectors is not None  # type narrowing

        if algorithm == "pca":
            # For PCA: project onto eigenvectors and scale
            W = scaling @ eigenvectors.T
        else:
            # For ZCA: project, scale, and rotate back
            W = eigenvectors @ scaling @ eigenvectors.T
    elif algorithm == "cholesky":
        # Based on Cholesky decomp, related to QR decomp
        covariance = stats.get("covariance", None)
        assert covariance is not None  # type narrowing
        L = torch.linalg.cholesky(covariance)
        Identity = torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
        # Solve L @ W = I for W, more stable and quicker than inv(L)
        W = torch.linalg.solve_triangular(L, Identity, upper=False)
    else:
        raise ValueError(f"Unknown whitening algorithm: {algorithm}, must be one of ['pca', 'zca', 'cholesky]")

    whitened = x_centered @ W.T

    if return_W:
        return whitened, W
    else:
        return whitened
