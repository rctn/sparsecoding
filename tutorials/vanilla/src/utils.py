"""Utils for vanilla sparse coding tutorial notebook."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import scipy.io as sio
from sparsecoding.transforms import sample_random_patches
import torch
from torchvision.utils import make_grid

SMOOTH_INTERVALS = 100


def show_patches(patches, title=""):
    """Given patches of images, create a grid and display it.

    patches is a Tensor with shape (batch_size, pixels_per_patch).
    """
    size = int(np.sqrt(patches.size(1)))
    batch_size = patches.size(0)
    img_grid = torch.reshape(patches, (-1, 1, size, size))
    out = make_grid(img_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(patches))[0]
    display(out, bar=False, title=title)


def show_patches_sbs(orig, recon, title="", dpi=200):
    """Given original and reconstructed patches, display grids side by side.

    orig and recon are Tensors each with shape (batch_size, pixels_per_patch).
    """
    size = int(np.sqrt(orig.size(1)))
    batch_size = orig.size(0)

    img_grid = torch.reshape(orig, (-1, 1, size, size))
    orig_out = make_grid(img_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(orig))[0]

    img_grid = torch.reshape(recon, (-1, 1, size, size))
    recon_out = make_grid(img_grid, padding=1, nrow=int(np.sqrt(batch_size)), pad_value=torch.min(recon))[0]

    display_sbs(orig_out, recon_out, bar=False, title=title, dpi=dpi)


def show_dict(dict, title):
    """Given basis vectors, create a grid and display it.

    dict is Tensor of (size_of_basis, num_bases).
    """
    size = int(np.sqrt(dict.size(0)))
    grid = torch.reshape(dict.T, (-1, 1, size, size))
    out = make_grid(grid, padding=1, nrow=8, pad_value=-1)[0]
    display(out, bar=False, title=title)


def show_imgs(imgs, dpi=200):
    """Show whitened images in a perceptually-friendly way.

    imgs is Tensor of shape (num_imgs, height, width).
    """
    grid = torch.unsqueeze(imgs, dim=1)
    out = make_grid(grid, padding=20, nrow=5, pad_value=-5)[0]
    display(out, bar=False, dpi=dpi, vrange=(-2, 2))


def display(img, title=None, bar=True, cmap="gray", dpi=150, vrange=None):
    """Display images in Jupyter notebook.

    Parameters
    ----------
    img : Tensor or NumPy array of image values to display.
    title: String; title of figure. Optional.
    bar : Whether to show color bar on the side. Optional; default True.
    cmap : Color map. Optional; default 'gray'.
    dpi : Controls size. Optional; default 150.
    vrange : Tuple (value_min, value_max). Optional (will set automatically).
    """
    plt.axis("off")
    if vrange:
        vmin = vrange[0]
        vmax = vrange[1]
    else:
        vmin = torch.min(img)
        vmax = torch.max(img)
    plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
    fig = plt.gcf()
    fig.set_dpi(dpi)
    if bar:
        plt.colorbar()
    if title:
        plt.title(title)
    plt.show()


def display_sbs(orig, recon, title=None, bar=True, cmap="gray", dpi=150):
    """Display two images side-by-side in Jupyter notebook.

    Parameters
    ----------
    orig : Tensor of original image patches to display. Shape is
        (batch_size, pixels_per_patch).
    recon : Tensor of reconstructed image patches to display. Shape is
        (batch_size, pixels_per_patch).
    title : String; title of figure. Optional.
    bar : Whether to show color bar on the side. Optional; default True.
    cmap : Color map. Optional; default 'gray'.
    dpi : Controls size. Optional; default 150.
    """

    plt.subplot(1, 2, 1)
    plt.title("original")
    plt.axis("off")
    plt.imshow(orig, cmap=cmap)
    if bar:
        plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("reconstructed")
    plt.axis("off")
    plt.imshow(recon, cmap=cmap)
    if bar:
        plt.colorbar()

    fig = plt.gcf()
    fig.set_dpi(dpi)
    if title:
        plt.suptitle(title)
    plt.show()


def create_patches(imgs, epochs, batch_size, N, rng):
    """Preprocesses patches.

    Parameters
    ----------
    imgs : Tensor of (num_imgs, height, width).
    epochs : Number of training epochs.
    batch_size : Batch size.
    N : Number of pixels per image.
    rng : PyTorch random number generator.

    Returns
    -------
    patches : Tensor of size (epochs, batch_size, pixels_per_patch).
    """
    # TODO: use rng here when sample_random_patches supports it.
    patches = sample_random_patches(int(np.sqrt(N)), batch_size * epochs, torch.unsqueeze(imgs, 1))
    patches = patches.reshape(epochs, batch_size, N)
    return patches


def load_data(img_path):
    """If whitened images have not been downloaded, download (~20MB).
    Also normalize images to mean 0 and variance 1.

    Returns
    -------
    imgs : Tensor of (num_imgs, height, width).
    """
    imgs = sio.loadmat(img_path)["IMAGES"]  # (512, 512, 10)
    # normalize to mean 0 var 1
    imgs = (imgs - np.mean(imgs, axis=(0, 1), keepdims=True)) / np.std(imgs, axis=(0, 1), keepdims=True)
    return torch.Tensor(imgs).permute(2, 0, 1)


def plot_loss(y, title):
    """Plots loss per epoch with smoothing. y is list of losses."""
    x = np.arange(len(y))
    spline = make_interp_spline(x, y)
    x_ = np.linspace(x.min(), x.max(), SMOOTH_INTERVALS)
    y_ = spline(x_)
    plt.plot(x_, y_)
    plt.title(title)
    plt.show()


def plot_loss_sbs(mse, sparse_cost):
    """Plots loss per epoch with smoothing.

    mse and sparse_cost are lists of scalars.
    """
    x_mse = np.arange(len(mse))
    spline_mse = make_interp_spline(x_mse, mse)
    xmse_ = np.linspace(x_mse.min(), x_mse.max(), SMOOTH_INTERVALS)
    ymse_ = spline_mse(xmse_)
    plt.subplot(1, 2, 1)
    plt.title("MSE")
    plt.plot(xmse_, ymse_)

    x_sc = np.arange(len(sparse_cost))
    spline_sc = make_interp_spline(x_sc, sparse_cost)
    xsc_ = np.linspace(x_sc.min(), x_sc.max(), SMOOTH_INTERVALS)
    ysc_ = spline_sc(xsc_)
    plt.subplot(1, 2, 2)
    plt.title("sparsity cost")
    plt.plot(xsc_, ysc_)

    fig = plt.gcf()
    fig.set_size_inches(20, 3)
    plt.show()


def plot_coeffs(coeffs, title="patch_coefficients"):
    """Plots coefficient stem plot for one patch.

    coeffs is Tensor of shape (batch_size, number_of_bases).
    """
    plt.stem(coeffs, use_line_collection=True)
    plt.title(title)
    plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    plt.show()


def coeff_grid(coeffs):
    """Plots stem for coefficients of half a batch, arranges them into a grid.

    coeffs is Tensor of shape (batch_size, number_of_bases).
    """
    batch_size = coeffs.shape[0]
    num_to_plot = batch_size // 2
    num_cols = 4

    fig = plt.figure()
    gs = fig.add_gridspec(num_to_plot // num_cols, num_cols, hspace=0, wspace=0)
    subplots = gs.subplots(sharex="col", sharey=True).flatten()
    for i, subplot in enumerate(subplots[:num_to_plot]):
        subplot.stem(coeffs[i], use_line_collection=True)

    fig.set_size_inches(30, 50)
    plt.show()


def show_components(phi, a, dpi):
    """Display weighted components sorted by coefficient size.

    phi is entire dictionary. a is coefficients of one patch.
    """
    patch_size = int(np.sqrt(phi.shape[0]))
    order = torch.flip(np.argsort(np.abs(a)), dims=[0])
    a = a[order].unsqueeze(1)
    phi = phi[:, order]

    weighted_phi = (phi * a.T).T
    weighted_phi = weighted_phi.reshape(-1, 1, patch_size, patch_size)
    components = make_grid(weighted_phi, ncol=int(np.sqrt(a.shape[0])), padding=1, pad_value=-1)[0]

    vmax = torch.max(weighted_phi)
    vmin = torch.min(weighted_phi)
    plt.axis("off")
    plt.imshow(components, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("weighted components")
    fig = plt.gcf()
    fig.set_dpi(dpi)
    plt.show()
