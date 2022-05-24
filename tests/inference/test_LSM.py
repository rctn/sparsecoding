import unittest

from sparsecoding import inference
from tests.testing_utilities import TestCase
from tests.inference.common import DATAS, DATASET, DICTIONARY


class TestLSM(TestCase):
    def test_shape(self):
        """
        Test that LSM inference returns expected shapes.
        """
        N_ITER = 10

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.LSM(N_ITER)
            a = inference_method.infer(data, DICTIONARY)
            self.assertShapeEqual(a, dataset.weights)

    def test_inference(self):
        """
        Test that LSM inference recovers the correct weights.
        """
        N_ITER = 1000

        for (data, dataset) in zip(DATAS, DATASET):
            inference_method = inference.LSM(n_iter=N_ITER)

            a = inference_method.infer(data, DICTIONARY)

            self.assertAllClose(a, dataset.weights, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
