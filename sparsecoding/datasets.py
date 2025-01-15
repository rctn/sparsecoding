import os

import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from sparsecoding.priors import Prior
from sparsecoding.transforms import patchify


class BarsDataset(Dataset):
    """Toy dataset where the dictionary elements are horizontal and vertical bars.

    Dataset elements are formed by taking linear combinations of the dictionary elements,
    where the weights are sampled according to the input Prior.

    Parameters
    ----------
    patch_size : int
        Side length for elements of the dataset.
    dataset_size : int
        Number of dataset elements to generate.
    prior : Prior
        Prior distribution on the weights. Should be sparse.

    Attributes
    ----------
    basis : Tensor, shape [2 * patch_size, patch_size, patch_size]
        Dictionary elements (horizontal and vertical bars).
    weights : Tensor, shape [dataset_size, 2 * patch_size]
        Weights for each of the dataset elements.
    data : Tensor, shape [dataset_size, patch_size, patch_size]
        Weighted linear combinations of the basis elements.
    """

    def __init__(
        self,
        patch_size: int,
        dataset_size: int,
        prior: Prior,
    ):
        self.P = patch_size
        self.N = dataset_size

        one_hots = torch.nn.functional.one_hot(torch.arange(self.P))  # [P, P]
        one_hots = one_hots.type(torch.float32)  # [P, P]

        h_bars = one_hots.reshape(self.P, self.P, 1)
        v_bars = one_hots.reshape(self.P, 1, self.P)

        h_bars = h_bars.expand(self.P, self.P, self.P)
        v_bars = v_bars.expand(self.P, self.P, self.P)
        self.basis = torch.cat((h_bars, v_bars), dim=0)  # [2*P, P, P]

        self.weights = prior.sample(self.N)  # [N, 2*P]

        self.data = torch.einsum(
            "nd,dhw->nhw",
            self.weights,
            self.basis,
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        return self.data[idx]


class FieldDataset(Dataset):
    """Dataset used in Olshausen & Field (1996).

    Paper:
        https://courses.cs.washington.edu/courses/cse528/11sp/Olshausen-nature-paper.pdf
        Emergence of simple-cell receptive field properties
        by learning a sparse code for natural images.

    Parameters
    ----------
    root : str
        Location to download the dataset to.
    patch_size : int
        Side length of patches for sparse dictionary learning.
    stride : int, optional
        Stride for sampling patches. If not specified, set to `patch_size`
        (non-overlapping patches).
    """

    B = 10
    C = 1
    H = 512
    W = 512

    def __init__(
        self,
        root: str,
        patch_size: int = 8,
        stride: int | None = None,
    ):
        self.P = patch_size
        if stride is None:
            stride = patch_size

        root = os.path.expanduser(root)
        os.system(f"mkdir -p {root}")
        if not os.path.exists(f"{root}/field.mat"):
            os.system("wget https://rctn.org/bruno/sparsenet/IMAGES.mat")
            os.system(f"mv IMAGES.mat {root}/field.mat")

        self.images = torch.tensor(loadmat(f"{root}/field.mat")["IMAGES"])  # [H, W, B]
        assert self.images.shape == (self.H, self.W, self.B)

        self.images = torch.permute(self.images, (2, 0, 1))  # [B, H, W]
        self.images = torch.reshape(self.images, (self.B, self.C, self.H, self.W))  # [B, C, H, W]

        self.patches = patchify(patch_size, self.images, stride)  # [B, N, C, P, P]
        self.patches = torch.reshape(self.patches, (-1, self.C, self.P, self.P))  # [B*N, C, P, P]

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return self.patches[idx]
