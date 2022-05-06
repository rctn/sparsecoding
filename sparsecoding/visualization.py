import numpy as np
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# TODO: Combine/refactor plot_dictionary and plot_patches; lots of repeated code.
# TODO: Add method for visualizing coefficients.
# TODO: Add method for visualizing reconstructions and original patches.
def plot_dictionary(dictionary,
                    color=False,
                    nrow=30,
                    normalize=True,
                    scale_each=True,
                    fig=None,
                    ax=None,
                    title='',
                    size=8):
    '''
    Parameters
    ----------
    dictionary : scalar (n_features,n_basis)

    color : boolean (1,) default=False
        set True if dictionary 3 channel (color)
    nrow : scalar (1,) default=30
        number of dictionary elements in a row
    normalize : boolean (1,) default=True
        normalize to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
    scale_each : boolean (1,) default=True
        scale each element to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
    fig : matplotlib.pyplot figure handle default=None
        if not provided, new handle created and returned
    ax : matplotlib.pyplot axes handle default=None
        if not provided, new handle created and returned
    title : string (1,) default=''
        title of plot
    size : scalar (1,) default=8
       plot size (inches)

    Returns
    -------
    fig : matplotlib.pyplot figure handle

    ax : matplotlib.pyplot axes handle
    '''

    n_features, n_basis = dictionary.shape

    nch = 1
    if color:
        nch = 3

    patch_size = int(np.sqrt(n_features//nch))

    D_imgs = dictionary.T.reshape([n_basis, patch_size, patch_size, nch]).permute([
        0, 3, 1, 2])  # swap channel dims for torch
    grid_img = torchvision.utils.make_grid(
        D_imgs, nrow=nrow, normalize=normalize, scale_each=scale_each).cpu()

    if fig == None or ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    ax.clear()
    ax.set_title(title)
    ax.imshow(grid_img.permute(1, 2, 0))  # swap channel dims for matplotlib
    ax.set_axis_off()
    fig.set_size_inches(size, size)
    fig.canvas.draw()
    return fig, ax


def plot_patches(patches,
                 color=False,
                 normalize=True,
                 scale_each=True,
                 fig=None,
                 ax=None,
                 title='',
                 size=8):
    '''
    Parameters
    ----------
    patches : scalar (batch_size, n_pixels)
    color : boolean (1,) default=False
        set True if dictionary 3 channel (color)
    normalize : boolean (1,) default=True
        normalize to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
    scale_each : boolean (1,) default=True
        scale each element to [0,1] (see https://pytorch.org/vision/main/generated/torchvision.utils.make_grid.html)
    fig : matplotlib.pyplot figure handle default=None
        if not provided, new handle created and returned
    ax : matplotlib.pyplot axes handle default=None
        if not provided, new handle created and returned
    title : string (1,) default=''
        title of plot
    size : scalar (1,) default=8
       plot size (inches)

    Returns
    -------
    fig : matplotlib.pyplot figure handle

    ax : matplotlib.pyplot axes handle
    '''

    batch_size = patches.shape[0]
    nrow = int(np.sqrt(patches.shape[0]))

    nch = 1
    if color:
        nch = 3

    patch_size = int(np.sqrt(patches.size(1)))

    D_imgs = patches.reshape(
        [batch_size, patch_size, patch_size, nch]).permute([
            0, 3, 1, 2])  # swap channel dims for torch
    grid_img = make_grid(
        D_imgs, nrow=nrow, normalize=normalize, scale_each=scale_each).cpu()

    if fig == None or ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    ax.clear()
    ax.set_title(title)
    ax.imshow(grid_img.permute(1, 2, 0))  # swap channel dims for matplotlib
    ax.set_axis_off()
    fig.set_size_inches(size, size)
    fig.canvas.draw()
    return fig, ax
