import unittest

from sparsecoding import inference
from tests.testing_utilities import TestCase
from tests.sparsecoding.inference.common import (
    DATAS, DATASET_SIZE, DATASET, DICTIONARY, PATCH_SIZE
)


class TestVanilla(TestCase):
    def test_shape(self):
        """
        Test that Vanilla inference returns expected shapes.
        """
        N_ITER = 10

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.Vanilla(N_ITER)
            a = inference_method.infer(data, DICTIONARY)
            self.assertShapeEqual(a, dataset.weights)

            inference_method = inference.Vanilla(N_ITER, return_all_coefficients=True)
            a = inference_method.infer(data, DICTIONARY)
            self.assertEqual(a.shape, (DATASET_SIZE, N_ITER + 1, 2 * PATCH_SIZE))

    def test_inference(self):
        """
        Test that Vanilla inference recovers the correct weights.
        """
        LR = 5e-2
        N_ITER = 1000

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.Vanilla(coeff_lr=LR, n_iter=N_ITER)

            a = inference_method.infer(data, DICTIONARY)

            self.assertAllClose(a, dataset.weights, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
