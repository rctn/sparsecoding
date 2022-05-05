from tests.testing_utilities import TestCase
import torch
import numpy as np


class TestTestCaseBaseClass(TestCase):

    def test_pytorch_all_close(self):
        result = torch.ones([10,10]) + 1e-10
        expected = torch.ones([10,10])
        self.assertAllClose(result, expected)
        
        
    def test_np_all_close(self):
        result = np.ones([100,100]) + 1e-10
        expected = np.ones([100,100])
        self.assertAllClose(result, expected)

        
    def test_assert_true(self):
        self.assertTrue(True==True)
        
        
    def test_assert_false(self):
        self.assertFalse(True!=True)
        
        
    def test_assert_equal(self):
        self.assertEqual('sparse coding','sparse coding')

        
    def test_assert_pytorch_shape_equal(self):
        a = torch.zeros([10,10]) 
        b = torch.ones([10,10])
        self.assertShapeEqual(a,b)
        
        
    def test_assert_np_shape_equal(self):
        a = np.zeros([100,100]) 
        b = np.ones([100,100])
        self.assertShapeEqual(a,b)
        