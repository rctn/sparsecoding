import torch
import unittest

from sparsecoding.data.datasets.bars import HierarchicalBarsDataset
from sparsecoding.models import Hierarchical
from sparsecoding.priors.laplace import LaplacePrior
from tests.testing_utilities import TestCase

torch.manual_seed(1997)

PATCH_SIZE = 8
DATASET_SIZE = 1000

PRIORS = [
    LaplacePrior(
        dim=6,
        scale=1.0,
        positive_only=False,
    ),
    LaplacePrior(
        dim=2 * PATCH_SIZE,
        scale=0.1,
        positive_only=False,
    ),
    LaplacePrior(
        dim=PATCH_SIZE * PATCH_SIZE,
        scale=0.01,
        positive_only=False,
    ),
]

DATASET = HierarchicalBarsDataset(
    patch_size=PATCH_SIZE,
    dataset_size=DATASET_SIZE,
    priors=PRIORS,
)


class TestHierarchical(TestCase):
    def test_infer_weights(self):
        """
        Test that Hierarchical inference recovers the correct weights.
        """
        model = Hierarchical(priors=PRIORS)
        model.bases = DATASET.bases

        weights = model.infer_weights(DATASET.data)

        inferred_log_probs = torch.mean(Hierarchical.log_prob(
            DATASET.data,
            DATASET.bases,
            PRIORS,
            weights[:-1],
        ))
        dataset_log_probs = torch.mean(Hierarchical.log_prob(
            DATASET.data,
            DATASET.bases,
            PRIORS,
            DATASET.weights[:-1],
        ))
        self.assertAllClose(inferred_log_probs, dataset_log_probs, atol=5e-2)

    def test_learn_bases(self):
        """
        Test that Hierarchical inference recovers the correct bases.
        """
        model = Hierarchical(priors=PRIORS)

        model.learn_bases(DATASET.data)

        weights = model.infer_weights(DATASET.data)

        inferred_log_probs = torch.mean(Hierarchical.log_prob(
            DATASET.data,
            list(map(lambda basis: basis.detach(), model.bases)),
            PRIORS,
            weights[:-1],
        ))
        dataset_log_probs = torch.mean(Hierarchical.log_prob(
            DATASET.data,
            DATASET.bases,
            PRIORS,
            DATASET.weights[:-1],
        ))
        self.assertAllClose(inferred_log_probs, dataset_log_probs, atol=3e0)


if __name__ == "__main__":
    unittest.main()
