import warnings

import torch


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

    h_patch_idxs, w_patch_idxs = torch.meshgrid(
        torch.arange(P),
        torch.arange(P),
        indexing='ij'
    )
    h_idxs = h_start_idx.reshape(N, 1, 1) + h_patch_idxs
    w_idxs = w_start_idx.reshape(N, 1, 1) + w_patch_idxs

    leading_idxs = [
        torch.randint(low=0, high=image.shape[d], size=(N, 1, 1))
        for d in range(image.dim() - 3)
    ]

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

    if (
        H % P != 0
        or W % P != 0
    ):
        warnings.warn(
            f"Image size ({H, W}) not evenly divisible by `patch_size` ({P}),"
            f"parts on the bottom and/or right will be cropped.",
            UserWarning,
        )

    N = (
        int((H - P + 1 + stride) // stride)
        * int((W - P + 1 + stride) // stride)
    )

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
        raise ValueError(
            f"Expected {N} patches per image, "
            f"got int(H/P) * int(W/P) = {int(H / P) * int(W / P)}."
        )

    if (
        H % P != 0
        or W % P != 0
    ):
        warnings.warn(
            f"Image size ({H, W}) not evenly divisible by `patch_size` ({P}),"
            f"parts on the bottom and/or right will be zeroed.",
            UserWarning,
        )

    patches = patches.reshape(-1, N, C*P*P)  # [prod(*), N, C*P*P]
    patches = torch.permute(patches, (0, 2, 1))  # [prod(*), C*P*P, N]
    image = torch.nn.functional.fold(
        input=patches,
        output_size=(H, W),
        kernel_size=P,
        stride=P,
    )  # [prod(*), C, H, W]

    return image.reshape(*leading_dims, C, H, W)
