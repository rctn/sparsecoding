import numpy as np
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# TODO: Combine/refactor plot_dictionary and plot_patches; lots of repeated code.
# TODO: Add method for visualizing coefficients.
# TODO: Add method for visualizing reconstructions and original patches.
def plot_dictionary(
    dictionary, color=False, nrow=30, normalize=True, scale_each=True, fig=None, ax=None, title="", size=8
):
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
        Normalize to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
    scale_each : bool, default=True
        Scale each element to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
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

    n_features, n_basis = dictionary.shape

    nch = 1
    if color:
        nch = 3

    patch_size = int(np.sqrt(n_features // nch))

    D_imgs = dictionary.T.reshape([n_basis, patch_size, patch_size, nch]).permute(
        [0, 3, 1, 2]
    )  # swap channel dims for torch
    grid_img = torchvision.utils.make_grid(D_imgs, nrow=nrow, normalize=normalize, scale_each=scale_each).cpu()

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    ax.clear()
    ax.set_title(title)
    ax.imshow(grid_img.permute(1, 2, 0))  # swap channel dims for matplotlib
    ax.set_axis_off()
    fig.set_size_inches(size, size)
    fig.canvas.draw()
    return fig, ax


def plot_patches(patches, color=False, normalize=True, scale_each=True, fig=None, ax=None, title="", size=8):
    """
    Parameters
    ----------
    patches : array-like, shape [batch_size, n_pixels]
        Image patches
    color : bool, default=False
        Set True if dictionary 3 channel (color)
    nrow : int, default=30
        Number of dictionary elements in a row
    normalize : bool, default=True
        Normalize to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
    scale_each : bool, default=True
        Scale each element to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
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

    batch_size = patches.shape[0]
    nrow = int(np.sqrt(patches.shape[0]))

    nch = 1
    if color:
        nch = 3

    patch_size = int(np.sqrt(patches.size(1)))

    D_imgs = patches.reshape([batch_size, patch_size, patch_size, nch]).permute(
        [0, 3, 1, 2]
    )  # swap channel dims for torch
    grid_img = make_grid(D_imgs, nrow=nrow, normalize=normalize, scale_each=scale_each).cpu()

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    ax.clear()
    ax.set_title(title)
    ax.imshow(grid_img.permute(1, 2, 0))  # swap channel dims for matplotlib
    ax.set_axis_off()
    fig.set_size_inches(size, size)
    fig.canvas.draw()
    return fig, ax
