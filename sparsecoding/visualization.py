import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# TODO: Add method for visualizing coefficients.


def _prepare_image_grid(data, batch_size, color=False, nrow=None):
    """Helper function to prepare image data for grid visualization.

    Parameters
    ----------
    data : array-like
        Input image data to prepare
    batch_size : int
        Number of images
    color : bool, default=False
        Set True if images are 3 channel (color)
    nrow : int, optional
        Number of images per row in grid

    Returns
    -------
    tensor
        Prepared image tensor ready for grid creation
    int
        Number of rows to use in grid
    """
    n_channels = 3 if color else 1
    patch_size = int(np.sqrt(data.size(1) // n_channels))

    if nrow is None:
        nrow = int(np.sqrt(batch_size))

    # Reshape and reorder dimensions for torch grid creation
    return (data.reshape(batch_size, patch_size, patch_size, n_channels)
            .permute(0, 3, 1, 2)), nrow


def plot_image_grid(data, color=False, nrow=None, normalize=True,
                    scale_each=True, fig=None, ax=None, title="", size=8):
    """Generic function to plot image data in a grid

    Parameters
    ----------
    data : array-like
        Input image data to visualize
    color : bool, default=False
        Set True if images are 3 channel (color)
    nrow : int, optional
        Number of images per row in grid
    normalize : bool, default=True
        Normalize to [0,1]
    scale_each : bool, default=True
        Scale each image to [0,1]
    fig : matplotlib.pyplot figure handle, optional
        If not provided, new handle created and returned
    ax : matplotlib.pyplot axes handle, optional
        If not provided, new handle created and returned
    title : str, optional
        Title of plot
    size : float, default=8
        Plot size (inches)

    Returns
    -------
    fig : matplotlib.pyplot figure handle
    ax : matplotlib.pyplot axes handle
    """
    batch_size = data.size(0)

    imgs, nrow = _prepare_image_grid(data, batch_size, color, nrow)

    grid_img = make_grid(
        imgs, nrow=nrow, normalize=normalize, scale_each=scale_each).cpu()

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    ax.clear()
    ax.set_title(title)
    ax.imshow(grid_img.permute(1, 2, 0))
    ax.set_axis_off()
    fig.set_size_inches(size, size)
    fig.canvas.draw()

    return fig, ax


def plot_dictionary(dictionary, color=False, nrow=30, normalize=True,
                    scale_each=True, fig=None, ax=None, title="", size=8):
    """Plot all elements of dictionary in grid

    Parameters
    ----------
    dictionary : array-like, shape [n_features, n_basis]
        Dictionary
    color : bool, default=False
        Set True if dictionary 3 channel (color)
    nrow : int, default=30
        Number of dictionary elements in a row
    normalize : bool, default=True
        Normalize to [0,1]
    scale_each : bool, default=True
        Scale each element to [0,1]
    fig : matplotlib.pyplot figure handle, optional
        If not provided, new handle created and returned
    ax : matplotlib.pyplot axes handle, optional
        If not provided, new handle created and returned
    title : str, optional
        Title of plot
    size : float, default=8
        Plot size (inches)

    Returns
    -------
    fig : matplotlib.pyplot figure handle
    ax : matplotlib.pyplot axes handle
    """
    return plot_image_grid(
        dictionary.T, color=color, nrow=nrow, normalize=normalize,
        scale_each=scale_each, fig=fig, ax=ax, title=title, size=size
    )


def plot_patches(patches, color=False, normalize=True, scale_each=True,
                 fig=None, ax=None, title="", size=8):
    """Plot image patches in grid

    Parameters
    ----------
    patches : array-like, shape [batch_size, n_pixels]
        Image patches
    color : bool, default=False
        Set True if patches are 3 channel (color)
    normalize : bool, default=True
        Normalize to [0,1]
    scale_each : bool, default=True
        Scale each patch to [0,1]
    fig : matplotlib.pyplot figure handle, optional
        If not provided, new handle created and returned
    ax : matplotlib.pyplot axes handle, optional
        If not provided, new handle created and returned
    title : str, optional
        Title of plot
    size : float, default=8
        Plot size (inches)

    Returns
    -------
    fig : matplotlib.pyplot figure handle
    ax : matplotlib.pyplot axes handle
    """
    return plot_image_grid(
        patches, color=color, normalize=normalize,
        scale_each=scale_each, fig=fig, ax=ax, title=title, size=size
    )


def plot_reconstructions(original_data, reconstructed_data, n_samples=None,
                         color=False, normalize=True, scale_each=True, size=12,
                         save_path=None):
    """Plot original data and reconstructions side by side for comparison

    Parameters
    ----------
    original_data : array-like, shape [batch_size, n_pixels]
        Original image patches
    reconstructed_data : array-like, shape [batch_size, n_pixels]
        Reconstructed image patches
    n_samples : int, optional
        Number of samples to display. If None, displays all samples
    color : bool, default=False
        Set True if patches are 3 channel (color)
    normalize : bool, default=True
        Normalize to [0,1]
    scale_each : bool, default=True
        Scale each patch to [0,1]
    size : float, default=12
        Base size for the plot (actual size will be size x size/2)
    save_path : str, optional
        If provided, save the plot to this path

    Returns
    -------
    fig : matplotlib.pyplot figure handle
        Only returned if save_path is None

    Raises
    ------
    ValueError
        If original_data and reconstructed_data have different shapes
    """
    if original_data.shape != reconstructed_data.shape:
        raise ValueError(
            f"Shape mismatch: original data shape {original_data.shape} "
            f"!= reconstruction shape {reconstructed_data.shape}"
        )

    total_samples = original_data.shape[0]
    if n_samples is None or n_samples > total_samples:
        n_samples = total_samples
    orig = original_data[:n_samples]
    recon = reconstructed_data[:n_samples]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size, size/2))

    # Plot original data
    plot_image_grid(
        orig, color=color, normalize=normalize, scale_each=scale_each,
        fig=fig, ax=ax1,
        title=f"Original (showing {n_samples} of {total_samples} samples)",
        size=size/2
    )

    # Plot reconstructions
    plot_image_grid(
        recon, color=color, normalize=normalize, scale_each=scale_each,
        fig=fig, ax=ax2,
        title=f"Reconstruction (showing {n_samples} of {total_samples} samples)",
        size=size/2
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig
