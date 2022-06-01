import os

from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

from sparsecoding.data.transforms.patch import patchify


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
    """

    B = 10
    C = 1
    H = 512
    W = 512

    def __init__(
        self,
        root: str,
        patch_size: int = 8,
    ):
        self.P = patch_size

        root = os.path.expanduser(root)
        os.system(f"mkdir -p {root}")
        if not os.path.exists(f"{root}/field.mat"):
            os.system("wget https://rctn.org/bruno/sparsenet/IMAGES.mat")
            os.system(f"mv IMAGES.mat {root}/field.mat")

        self.images = torch.tensor(loadmat(f"{root}/field.mat")["IMAGES"])  # [H, W, B]
        assert self.images.shape == (self.H, self.W, self.B)

        self.images = torch.permute(self.images, (2, 0, 1))  # [B, H, W]
        self.images = torch.reshape(self.images, (self.B, self.C, self.H, self.W))  # [B, C, H, W]

        self.patches = patchify(patch_size, self.images)  # [B, N, C, P, P]
        self.patches = torch.reshape(self.patches, (-1, self.C, self.P, self.P))  # [B*N, C, P, P]

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return self.patches[idx]
