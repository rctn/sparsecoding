import unittest

from sparsecoding import inference
from tests.testing_utilities import TestCase
from tests.inference.common import (
    DATAS, DATASET_SIZE, DATASET, DICTIONARY, PATCH_SIZE
)


class TestCEL0(TestCase):
    def test_shape(self):
        """
        Test that CEL0 inference returns expected shapes.
        """
        N_ITER = 10

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.CEL0(N_ITER)
            a = inference_method.infer(data, DICTIONARY)
            self.assertShapeEqual(a, dataset.weights)

            inference_method = inference.CEL0(N_ITER, return_all_coefficients=True)
            a = inference_method.infer(data, DICTIONARY)
            self.assertEqual(a.shape, (DATASET_SIZE, N_ITER + 1, 2 * PATCH_SIZE))

    def test_inference(self):
        """
        Test that CEL0 inference recovers the correct weights.
        """
        N_ITER = 1000

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.CEL0(n_iter=N_ITER, coeff_lr=1e-1, threshold=5e-1)

            a = inference_method.infer(data, DICTIONARY)

            self.assertAllClose(a, dataset.weights, atol=5e-2, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
