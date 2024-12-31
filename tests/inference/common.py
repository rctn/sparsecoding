import torch

from sparsecoding.datasets import BarsDataset
from sparsecoding.priors import L0Prior, SpikeSlabPrior

torch.manual_seed(1997)

PATCH_SIZE = 8
DATASET_SIZE = 1000

PRIORS = [
    SpikeSlabPrior(
        dim=2 * PATCH_SIZE,
        p_spike=0.8,
        scale=1.0,
        positive_only=True,
    ),
    L0Prior(
        prob_distr=(
            torch.nn.functional.one_hot(
                torch.tensor(1),
                num_classes=2 * PATCH_SIZE,
            ).type(torch.float32)
        ),
    ),
]

DATASET = [
    BarsDataset(
        patch_size=PATCH_SIZE,
        dataset_size=DATASET_SIZE,
        prior=prior,
    )
    for prior in PRIORS
]

DATAS = [
    dataset.data.reshape((DATASET_SIZE, PATCH_SIZE * PATCH_SIZE))
    for dataset in DATASET
]
DICTIONARY = DATASET[0].basis.reshape((2 * PATCH_SIZE, PATCH_SIZE * PATCH_SIZE)).T
