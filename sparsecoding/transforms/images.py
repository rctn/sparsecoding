import torch
import torch.fft as fft
import warnings
from typing import Dict, Optional
from functools import lru_cache
from .whiten import whiten, compute_whitening_stats


def check_images(images: torch.Tensor, algorithm: str = "zca"):
    """Verify that tensor is in the shape [N, C, H, W] and C != when using fourier based method"""

    if len(images.shape) != 4:
        raise ValueError("Images must be in shape [N, C, H, W]")

    if images.shape[1] != 1 and algorithm == "frequency":
        raise ValueError(
            "When using frequency based decorrelation, images must"
            + f"be grayscale, received {images.shape[1]} channels"
        )

    # Running cov based methods on large images can eat memory
    if algorithm in ["zca", "pca", "cholesky"] and (images.shape[2] > 64 or images.shape[3] > 64):
        print(
            f"WARNING: Running covaraince based whitening for images of size {images.shape[2]}x{images.shape[3]}."
            + "It is not recommended to use this for images smaller than 64x64"
        )

    # Running cov based methods on large images can eat memory
    if algorithm == "frequency" and (images.shape[2] <= 64 or images.shape[3] <= 64):
        print(
            f"WARNING: Running frequency based whitening for images of size {images.shape[2]}x{images.shape[3]}."
            + "It is recommended to use this for images larger than 64x64"
        )


def whiten_images(images: torch.Tensor, algorithm: str, stats: Dict = None, **kwargs) -> torch.Tensor:
    """
    Wrapper for all whitening transformations

    Parameters
    ----------
    images: tensor of shape (N, C, H, W)
    algorithm: what whitening transform we want to use
    stats: dictionary of dataset statistics needed for whitening transformations

    Returns
    ----------
    Tensor of whitened data in shape (N, C, H, W)
    """

    check_images(images, algorithm)

    if algorithm == "frequency":
        return frequency_whitening(images, **kwargs)

    elif algorithm in ["zca", "pca", "cholesky"]:
        N, C, H, W = images.shape
        flattened_images = images.flatten(start_dim=1)
        whitened = whiten(flattened_images, algorithm, stats, **kwargs)
        return whitened.reshape((N, C, H, W))

    else:
        raise ValueError(
            f"Unknown whitening algorithm: {algorithm}, \
                          must be one of ['frequency', 'pca', 'zca', 'cholesky]"
        )


def compute_image_whitening_stats(images: torch.Tensor) -> Dict:
    """
    Wrapper for computing whitening stats of an image dataset

    Parameters
    ----------
    images: tensor of shape (N, C, H, W)
    n_components: Number of principal components to keep. If None, keep all components.
                  If int, keep that many components. If float between 0 and 1,
                  keep components that explain that fraction of variance.

    Returns
    ----------
    Dictionary containing whitening statistics (eigenvalues, eigenvectors, mean)
    """
    check_images(images)
    flattened_images = images.flatten(start_dim=1)
    return compute_whitening_stats(flattened_images)


def create_frequency_filter(image_size: int, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Create a frequency domain filter for image whitening.

    Parameters
    ----------
    image_size: Size of the square image
    f0_factor: Factor for determining the cutoff frequency (default 0.4)

    Returns
    ----------
    torch.Tensor: Frequency domain filter
    """
    fx = torch.linspace(-image_size / 2, image_size / 2 - 1, image_size)
    fy = torch.linspace(-image_size / 2, image_size / 2 - 1, image_size)
    fx, fy = torch.meshgrid(fx, fy, indexing="xy")

    rho = torch.sqrt(fx**2 + fy**2)
    f_0 = f0_factor * image_size
    filt = rho * torch.exp(-((rho / f_0) ** 4))

    return fft.fftshift(filt)


@lru_cache(maxsize=32)
def get_cached_filter(image_size: int, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Get a cached frequency filter for the given image size.

    Parameters
    ----------
    image_size: Size of the square image
    f0_factor: Factor for determining the cutoff frequency

    Returns
    ----------
    torch.Tensor: Cached frequency domain filter
    """
    return create_frequency_filter(image_size, f0_factor)


def normalize_variance(tensor: torch.Tensor, target_variance: float = 1.0) -> torch.Tensor:
    """
    Normalize the variance of a tensor to a target value.

    Parameters
    ----------
    tensor: Input tensor
    target_variance: Desired variance after normalization

    Returns
    ----------
    torch.Tensor: Normalized tensor
    """

    centered = tensor - tensor.mean()
    current_variance = torch.var(centered)

    if current_variance > 0:
        scale_factor = torch.sqrt(torch.tensor(target_variance) / current_variance)
        return centered * scale_factor
    return centered


def whiten_channel(channel: torch.Tensor, filt: torch.Tensor, target_variance: float = 1.0) -> torch.Tensor:
    """
    Apply frequency domain whitening to a single channel.

    Parameters
    ----------
    channel: Single channel image tensor
    filt: Frequency domain filter
    target_variance: Target variance for normalization

    Returns
    ----------
    torch.Tensor: Whitened channel
    """

    if torch.var(channel) < 1e-8:
        return channel

    # Convert to frequency domain and apply filter
    If = fft.fft2(channel)
    If_whitened = If * filt.to(channel.device)

    # Convert back to spatial domain and normalize
    whitened = torch.real(fft.ifft2(If_whitened))

    # Normalize variance
    whitened = normalize_variance(whitened, target_variance)

    return whitened


def frequency_whitening(images: torch.Tensor, target_variance: float = 0.1, f0_factor: float = 0.4) -> torch.Tensor:
    """
    Apply frequency domain decorrelation to batched images.
    Method used in original sparsenet in Olshausen and Field in Nature
    and http://www.rctn.org/bruno/sparsenet/

    Parameters
    ----------
    images: Input images of shape (N, C, H, W)
    target_variance: Target variance for normalization
    f0_factor: Factor for determining filter cutoff frequency

    Returns
    ----------
    torch.Tensor: Whitened images
    """
    _, _, H, W = images.shape
    if H != W:
        raise ValueError("Images must be square")

    # Get cached filter
    filt = get_cached_filter(H, f0_factor)

    # Process each image in the batch
    whitened_batch = []
    for img in images:
        whitened_batch.append(whiten_channel(img[0], filt, target_variance))

    return torch.stack(whitened_batch).unsqueeze(1)


class WhiteningTransform(object):
    """
    A PyTorch transform for image whitening that can be used in a transform pipeline.
    Supports frequency, PCA, and ZCA whitening methods.
    """

    def __init__(self, algorithm: str = "zca", stats: Optional[Dict] = None, compute_stats: bool = False, **kwargs):
        """
        Initialize whitening transform.

        Parameters
        ----------
        algorithm: One of ['frequency', 'pca', 'zca', 'cholesky]
        stats: Pre-computed statistics for PCA/ZCA whitening
        compute_stats: If True, will compute stats on first batch seen
        **kwargs: Additional arguments passed to whitening function
        """
        self.algorithm = algorithm
        self.stats = stats
        self.compute_stats = compute_stats
        self.kwargs = kwargs

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply whitening transform to images.

        Parameters
        ----------
            images: Input images of shape [N, C, H, W] or [C, H, W]

        Returns
        ----------
            Whitened images of same shape as input
        """
        # Add batch dimension if necessary
        if images.dim() == 3:
            images = images.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        check_images(images)
        # Apply whitening
        whitened = whiten_images(images, self.algorithm, self.stats, **self.kwargs)

        # Remove batch dimension if input was single image
        if single_image:
            whitened = whitened.squeeze(0)

        return whitened

    def __repr__(self):
        return "custom whitening augmentation"


def sample_random_patches(
    patch_size: int,
    num_patches: int,
    image: torch.Tensor,
):
    """Sample random patches from an image.

    Parameters
    ----------
    patch_size : int
        Patch side length.
    num_patches : int
        Number of patches to sample.
    image : Tensor, shape [*, C, H, W]
        where:
            C is the number of channels,
            H is the image height,
            W is the image width.

    Returns
    -------
    patches : Tensor, shape [num_patches, C, patch_size, patch_size]
        Sampled patches from the input image(s).
    """
    P = patch_size
    N = num_patches
    H, W = image.shape[-2:]

    h_start_idx = torch.randint(
        low=0,
        high=H - P + 1,
        size=(N,),
    )
    w_start_idx = torch.randint(
        low=0,
        high=W - P + 1,
        size=(N,),
    )

    h_patch_idxs, w_patch_idxs = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    h_idxs = h_start_idx.reshape(N, 1, 1) + h_patch_idxs
    w_idxs = w_start_idx.reshape(N, 1, 1) + w_patch_idxs

    leading_idxs = [torch.randint(low=0, high=image.shape[d], size=(N, 1, 1)) for d in range(image.dim() - 3)]

    idxs = leading_idxs + [slice(None), h_idxs, w_idxs]

    patches = image[idxs]  # [N, P, P, C]

    return torch.permute(patches, (0, 3, 1, 2))


def patchify(
    patch_size: int,
    image: torch.Tensor,
    stride: int = None,
):
    """Break an image into square patches.

    Inverse of `quilt()`.

    Parameters
    ----------
    patch_size : int
        Patch side length.
    image : Tensor, shape [*, C, H, W]
        where:
            C is the number of channels,
            H is the image height,
            W is the image width.
    stride : int, optional
        Stride between patches in pixel space. If not specified, set to
        `patch_size` (non-overlapping patches).

    Returns
    -------
    patches : Tensor, shape [*, N, C, P, P]
        Non-overlapping patches taken from the input image,
        where:
            P is the patch size,
            N is the number of patches, equal to H//P * W//P,
            C is the number of channels of the input image.
    """
    leading_dims = image.shape[:-3]
    C, H, W = image.shape[-3:]
    P = patch_size
    if stride is None:
        stride = P

    if H % P != 0 or W % P != 0:
        warnings.warn(
            f"Image size ({H, W}) not evenly divisible by `patch_size` ({P}),"
            f"parts on the bottom and/or right will be cropped.",
            UserWarning,
        )

    N = int((H - P + 1 + stride) // stride) * int((W - P + 1 + stride) // stride)

    patches = torch.nn.functional.unfold(
        input=image.reshape(-1, C, H, W),
        kernel_size=P,
        stride=stride,
    )  # [prod(*), C*P*P, N]
    patches = torch.permute(patches, (0, 2, 1))  # [prod(*), N, C*P*P]

    assert patches.shape[1] == N

    return patches.reshape(*leading_dims, N, C, P, P)


def quilt(
    height: int,
    width: int,
    patches: torch.Tensor,
):
    """Gather square patches into an image.

    Inverse of `patchify()`.

    Parameters
    ----------
    height : int
        Height for the reconstructed image.
    width : int
        Width for the reconstructed image.
    patches : Tensor, shape [*, N, C, P, P]
        Non-overlapping patches from an input image,
        where:
            P is the patch size,
            N is the number of patches,
            C is the number of channels in the image.

    Returns
    -------
    image : Tensor, shape [*, C, height, width]
        Image reconstructed by stitching together input patches.
    """
    leading_dims = patches.shape[:-4]
    N, C, P = patches.shape[-4:-1]
    H = height
    W = width

    if int(H / P) * int(W / P) != N:
        raise ValueError(f"Expected {N} patches per image, " f"got int(H/P) * int(W/P) = {int(H / P) * int(W / P)}.")

    if H % P != 0 or W % P != 0:
        warnings.warn(
            f"Image size ({H, W}) not evenly divisible by `patch_size` ({P}),"
            f"parts on the bottom and/or right will be zeroed.",
            UserWarning,
        )

    patches = patches.reshape(-1, N, C * P * P)  # [prod(*), N, C*P*P]
    patches = torch.permute(patches, (0, 2, 1))  # [prod(*), C*P*P, N]
    image = torch.nn.functional.fold(
        input=patches,
        output_size=(H, W),
        kernel_size=P,
        stride=P,
    )  # [prod(*), C, H, W]

    return image.reshape(*leading_dims, C, H, W)
