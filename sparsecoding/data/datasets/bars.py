import numpy as np
import torch
from torch.utils.data import Dataset

from sparsecoding.priors.common import Prior


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