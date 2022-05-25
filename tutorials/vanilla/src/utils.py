"""Utils for vanilla sparse coding tutorial notebook."""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from requests import patch
from scipy.interpolate import make_interp_spline
import scipy.io as sio
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import urllib.request

SMOOTH_INTERVALS = 100


def visualize_patches(patches, title=""):
    """
    Given patches of images in the dataset, create a grid and display it.

    Parameters
    ----------
    patches: Tensor of (batch_size, pixels_per_patch).
    title: String; title of figure. Optional.
    """
    size = int(np.sqrt(patches.size(1)))
    batch_size = patches.size(0)
    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(patches[i, :], (1, size, size))
        img_grid.append(img)

    out = make_grid(img_grid, padding=1, nrow=int(
        np.sqrt(batch_size)), pad_value=torch.min(patches))[0, :, :]
    display(out, bar=False, title=title)


# Side by side.
def visualize_patches_sbs(orig, recon, title=""):
    """Given patches of images in the dataset, create a grid and display it.

    Parameters
    ----------
    patches: Tensor of (batch_size, pixels_per_patch).
    title: String; title of figure. Optional.
    """
    size = int(np.sqrt(orig.size(1)))
    batch_size = orig.size(0)

    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(orig[i, :], (1, size, size))
        img_grid.append(img)
    orig_out = make_grid(img_grid, padding=1, nrow=int(
        np.sqrt(batch_size)), pad_value=torch.min(orig))[0, :, :]

    img_grid = []
    for i in range(batch_size):
        img = torch.reshape(recon[i, :], (1, size, size))
        img_grid.append(img)
    recon_out = make_grid(img_grid, padding=1, nrow=int(
        np.sqrt(batch_size)), pad_value=torch.min(recon))[0, :, :]

    display2(orig_out, recon_out, bar=False, title=title, dpi=200)


def visualize_bases(bases, title):
    """
    Given basis vectors, create a grid and display it.

    Parameters
    ----------
    patches: Tensor of (N, num_bases).
    title: String; title of figure. Optional.
    """
    size = int(np.sqrt(bases.size(0)))
    grid = []
    for i in range(bases.size(0)):
        grid.append(torch.reshape(bases[:, i], (1, size, size)))
    out = make_grid(grid, padding=1, nrow=8, pad_value=-1)[0, :, :]
    display(out, bar=False, title=title)


def visualize_imgs(imgs):
    """
    Visualize whitened images in a perceptually-friendly way.

    Parameters
    ----------
    imgs: Tensor of (height, width, num_imgs).
    """
    grid = []
    for i in range(imgs.shape[-1]):
        reshaped = torch.reshape(torch.Tensor(imgs[:, :, i]), (1, 512, 512))
        grid.append(reshaped)
    out = make_grid(grid, padding=20, nrow=5, pad_value=-5)[0, :, :]
    display(out, bar=False, dpi=200, vrange=(-2, 2))


def display2(orig, recon, title=None, bar=True, cmap="gray", dpi=150, vrange=None):
    """Display two images side-by-side in Jupyter notebook.

    Parameters
    ----------
    img1: Tensor or NumPy array of image values to display.
    img2: Tensor or NumPy array of image values to display.
    title: String; title of figure. Optional.
    bar: Whether to show color bar on the side. Optional; default True.
    cmap: Color map. Optional; default 'gray'.
    dpi: Controls size. Optional; default 150.
    vrange: Tuple (value_min, value_max). Optional (will set automatically).
    """
    if vrange:
        vmin = vrange[0]
        vmax = vrange[1]
    else:
        vmin = torch.min(torch.min(orig), torch.min(recon))
        vmax = torch.max(torch.max(orig), torch.max(recon))

    plt.subplot(1, 2, 1)
    plt.title("original")
    plt.axis("off")
    # plt.imshow(orig, cmap=cmap)
    plt.imshow(orig, cmap=cmap, vmin=vmin, vmax=vmax)
    if bar:
        plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("reconstructed")
    plt.axis("off")
    # plt.imshow(orig, cmap=cmap)
    plt.imshow(orig, cmap=cmap, vmin=vmin, vmax=vmax)
    if bar:
        plt.colorbar()

    fig = plt.gcf()
    fig.set_dpi(dpi)
    if title:
        plt.suptitle(title)
    plt.show()


def display(img, title=None, bar=True, cmap="gray", dpi=150, vrange=None):
    """Display images in Jupyter notebook.

    Parameters
    ----------
    img: Tensor or NumPy array of image values to display.
    title: String; title of figure. Optional.
    bar: Whether to show color bar on the side. Optional; default True.
    cmap: Color map. Optional; default 'gray'.
    dpi: Controls size. Optional; default 150.
    vrange: Tuple (value_min, value_max). Optional (will set automatically).
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


def extract_patches(imgs, patch_size, batch_size, rng):
    """Want 64xbs patch mtx, so one patch is 8x8 patch of 512."""
    img_size = imgs.shape[0]
    img_idx = rng.integers(low=0, high=imgs.shape[2], size=batch_size)
    batch = imgs[:, :, img_idx]

    # get random upper left coord of patch
    start = rng.integers(low=0, high=img_size -
                         patch_size, size=2*batch_size)
    start = start.reshape(2, batch_size)

    # get batch_size random 8x8 patches
    patches = np.zeros((batch_size, patch_size, patch_size))
    for i in range(batch_size):
        patches[i, :, :] = batch[
            start[0, i]:start[0, i]+patch_size, start[1, i]:start[1, i]+patch_size, i]

    return patches.reshape((batch_size, patch_size*patch_size))


def create_patches(imgs, epochs, batch_size, N, patch_dir, rng):
    """Preprocess or load patches for faster training, ~100MB.

    Parameters
    ----------
    imgs: Tensor of (height, width, num_imgs).
    epochs: Number of training epochs.
    batch_size: Batch size.
    N: Number of basis functions.
    patch_dir: Path at which to store patches.

    Returns
    -------
    patches: Tensor of (batch_size, pixels_per_patch).
    """
    patch_size = int(np.sqrt(N))
    if patch_dir and os.path.exists(patch_dir):
        print("loading patches already preprocessed at {}".format(patch_dir))
        patches = torch.load(patch_dir)
    else:
        print("preprocessing patches; this will take a few minutes")
        patches = torch.zeros((epochs, batch_size, N))
        for i in tqdm(range(epochs)):
            patches[i, :, :] = torch.Tensor(
                extract_patches(imgs, patch_size, batch_size, rng))
        torch.save(patches, patch_dir)
    return patches


# def maybe_download_data(img_path):
#     """If whitened images have not been downloaded, download (~20MB).

#     Returns
#     -------
#     imgs: Tensor of (height, width, num_imgs).
#     """
#     if not os.path.exists(img_path):
#         cwd = os.getcwd()
#         print("downloading data to {}".format(os.path.join(cwd, img_path)))
#         data_url = "http://rctn.org/bruno/data/IMAGES.mat"
#         urllib.request.urlretrieve(data_url, img_path)
#     else:
#         print("data exists; not downloading")
#     # 512x512x10 (10 512x512 whitened images)
#     imgs = sio.loadmat(img_path)["IMAGES"]
#     return imgs


def load_data(img_path):
    """If whitened images have not been downloaded, download (~20MB).

    Returns
    -------
    imgs: Tensor of (height, width, num_imgs).
    """
    imgs = sio.loadmat(img_path)["IMAGES"]
    visualize_imgs(imgs)
    return imgs


def plot_loss(y, title):
    """Plots loss per epoch with smoothing."""
    x = np.arange(len(y))
    spline = make_interp_spline(x, y)
    x_ = np.linspace(x.min(), x.max(), SMOOTH_INTERVALS)
    y_ = spline(x_)
    plt.plot(x_, y_)
    plt.title(title)
    plt.show()


def plot_loss_sbs(mse, sparse_cost, dpi=100):
    """Plots loss per epoch with smoothing."""
    x_mse = np.arange(len(mse))
    spline_mse = make_interp_spline(x_mse, mse)
    xmse_ = np.linspace(x_mse.min(), x_mse.max(), SMOOTH_INTERVALS)
    ymse_ = spline_mse(xmse_)
    plt.subplot(1, 2, 1)
    plt.title("MSE")
    # plt.axis("off")
    plt.plot(xmse_, ymse_)

    x_sc = np.arange(len(sparse_cost))
    spline_sc = make_interp_spline(x_sc, sparse_cost)
    xsc_ = np.linspace(x_sc.min(), x_sc.max(), SMOOTH_INTERVALS)
    ysc_ = spline_sc(xsc_)
    plt.subplot(1, 2, 2)
    plt.title("sparsity cost")
    # plt.axis("off")
    plt.plot(xsc_, ysc_)

    fig = plt.gcf()
    # fig.set_dpi(dpi)
    fig.set_size_inches(20, 3)
    plt.show()


def plot_coeffs(coeffs, title="patch_coefficients"):
    """Plots coefficients for one image patch in dataset."""
    plt.stem(coeffs, use_line_collection=True)
    plt.title(title)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.show()


def coeff_grid(coeffs):
    """Plots stem plots of all coefficients, arranges them into a grid."""
    batch_size = coeffs.shape[0]
    dim = int(np.sqrt(batch_size))
    fig = plt.gcf()
    for i in range(dim**2):
        plt.subplot(batch_size//5, 5, i+1)
        plt.stem(coeffs[i], use_line_collection=True)
        plt.title("patch {}".format(i))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            labelbottom=False)  # labels along the bottom edge are off
    fig.set_size_inches(30, 100)
    plt.show()


# def coeff_grid2(coeffs, patches):
#     """Plots stem plots of all coefficients, arranges them into a grid."""
#     batch_size = coeffs.shape[0]
#     M = int(np.sqrt(coeffs.shape[1]))
#     dim = int(np.sqrt(batch_size))
#     # fig = plt.gcf()
#     fig = plt.figure()

#     # gs = gridspec.GridSpec(batch_size//5, 5, hspace=0, wspace=0)
#     gs = fig.add_gridspec(batch_size//5, 5, hspace=0, wspace=0)
#     # subplots = gs.subplots(sharex=True, sharey=True)
#     subplots = gs.subplots()
#     # fig.suptitle("coefficients per patch")
#     for i, subplot in enumerate(subplots.flatten()):
#         subplot.stem(coeffs[i])
#         # patch_subplot = subplot.subgridspec(2, 8)
#         # (ax3a, ax3b) = patch_subplot.subplots()

#     # for i in range(batch_size//5):
#     #     for j in range(5):
#     #         inner_grid = gs[i, j].subgridspec(2, 1, wspace=0)
#     #         (ax3a, ax3b) = inner_grid.subplots()

#     fig.set_size_inches(30, 80)
#     plt.show()
#     # plt.stem(coeffs[i], use_line_collection=True)
#     # subplot.title("patch {}".format(i))


def show_components(phi, a, tol=1e-3):
    """Inputs are entire phi, a for one patch."""
    patch_size = int(np.sqrt(a.shape[0]))
    # nonzeros = np.where(abs(a) > tol)[0]
    nonzeros = np.array(range(a.shape[0]))  # Hack to show all phi.
    relevant_phis = np.squeeze(phi[:, nonzeros]).T
    relevant_as = a[nonzeros]
    order = torch.flip(np.argsort(np.abs(relevant_as)), dims=[0])
    relevant_as = relevant_as[order]
    relevant_phis = relevant_phis[order]

    # why doesn't this work
    # weighted_phi = relevant_phis * a[nonzeros]
    # weighted_phi = weighted_phi.reshape(
    #     len(nonzeros), 1, patch_size, patch_size).permute(0, 1, 3, 2)
    # print(weighted_phi.shape)
    # vmax = torch.max(weighted_phi)
    # vmin = torch.min(weighted_phi)

    grid = []
    vmax = 0
    vmin = np.inf
    for i in range(relevant_phis.shape[0]):
        weighted_phi = relevant_phis[i, :] * relevant_as[i]
        if torch.min(weighted_phi) < vmin:
            vmin = torch.min(weighted_phi)
        if torch.max(weighted_phi) > vmax:
            vmax = torch.max(weighted_phi)
        grid.append(weighted_phi.reshape(1, patch_size, patch_size))
    components = make_grid(grid, ncol=int(
        np.sqrt(len(nonzeros))), padding=1, pad_value=-1, vrange=(vmin, vmax))[0]
    plt.axis("off")
    plt.imshow(components, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("weighted components")
    fig = plt.gcf()
    fig.set_dpi(100)
    plt.show()
