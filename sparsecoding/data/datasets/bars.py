import itertools
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from sparsecoding.priors.common import Prior
from sparsecoding.models import Hierarchical


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
        self.basis /= np.sqrt(self.P)  # Normalize basis.

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


class HierarchicalBarsDataset(Dataset):
    """Toy hierarchical dataset of horizontal and vertical bars.

    The L=1 basis functions are horizontal and vertical bars.

    The L=0 basis functions are equal mixtures of two horizontal and vertical bars
    on the image border.

    Parameters
    ----------
    patch_size : int
        Side length for elements of the dataset.
    dataset_size : int
        Number of dataset elements to generate.
    priors : List[Prior]
        Prior distributions on the weights, starting from the top-level basis
        and going down.

    Attributes
    ----------
    bases : List[Tensor],
        shapes:
            - [6, 2 * patch_size]
            - [2 * patch_size, patch_size * patch_size]
        Dictionary elements (combinations of horizontal and vertical bars).
    weights : List[Tensor],
        shapes:
            - [dataset_size, 6],
            - [dataset_size, 2 * patch_size],
            - [dataset_size, patch_size * patch_size].
        Weights for each level of the hierarchy.
    data : Tensor, shape [dataset_size, patch_size * patch_size]
        Weighted linear combinations of the basis elements.
    """

    def __init__(
        self,
        patch_size: int,
        dataset_size: int,
        priors: List[Prior],
    ):
        self.P = patch_size
        self.N = dataset_size
        self.priors = priors

        # Specify l1_basis: bars.
        one_hots = torch.nn.functional.one_hot(torch.arange(self.P))  # [P, P]
        one_hots = one_hots.type(torch.float32)  # [P, P]

        h_bars = one_hots.reshape(self.P, self.P, 1)
        v_bars = one_hots.reshape(self.P, 1, self.P)

        h_bars = h_bars.expand(self.P, self.P, self.P)
        v_bars = v_bars.expand(self.P, self.P, self.P)
        l1_basis = torch.cat((h_bars, v_bars), dim=0)  # [2*P, P, P]
        l1_basis /= np.sqrt(self.P)  # Normalize basis.
        l1_basis = l1_basis.reshape((2 * self.P, self.P * self.P))

        # Specify l0_basis: combinations of two bars on the border.
        border_bar_idxs = [0, self.P - 1, self.P, 2 * self.P - 1]
        l0_basis_idxs = torch.tensor(list(itertools.combinations(border_bar_idxs, 2)))
        l0_basis = torch.zeros((6, 2 * self.P), dtype=torch.float32)
        l0_basis[torch.arange(6), l0_basis_idxs[:, 0]] = 1. / np.sqrt(2.)
        l0_basis[torch.arange(6), l0_basis_idxs[:, 1]] = 1. / np.sqrt(2.)

        self.bases = [l0_basis, l1_basis]

        self.weights = list(map(
            lambda prior: prior.sample(self.N),
            self.priors,
        ))

        self.data = Hierarchical.generate(self.bases, self.weights)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        return self.data[idx]
