import torch
import unittest

from sparsecoding import inference
from tests.testing_utilities import TestCase
from tests.inference.common import (
    DATAS, DATASET_SIZE, DATASET, DICTIONARY, PATCH_SIZE
)


class TestLCA(TestCase):
    def test_shape(self):
        """
        Test that LCA inference returns expected shapes.
        """
        N_ITER = 10

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.LCA(N_ITER)
            a = inference_method.infer(data, DICTIONARY)
            self.assertShapeEqual(a, dataset.weights)

            for retval in ["active", "membrane"]:
                inference_method = inference.LCA(N_ITER, return_all_coefficients=retval)
                a = inference_method.infer(data, DICTIONARY)
                self.assertEqual(a.shape, (DATASET_SIZE, N_ITER + 1, 2 * PATCH_SIZE))

    def test_inference(self):
        """
        Test that LCA inference recovers the correct weights.
        """
        LR = 5e-2
        THRESHOLD = 0.1
        N_ITER = 1000

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.LCA(
                coeff_lr=LR,
                threshold=THRESHOLD,
                n_iter=N_ITER,
            )

            a = inference_method.infer(data, DICTIONARY)

            self.assertAllClose(a, dataset.weights, atol=5e-2)

if __name__ == "__main__":
    unittest.main()
