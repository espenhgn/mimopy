#!/usr/bin/env python
'''testing module for mimopy package'''
import numpy as np
import unittest
from .mimo import MIMO

class TestMIMO(unittest.TestCase):
    '''
    A set of test functions for the MIMO class.
    '''
    def test_01(self):
        x = np.random.rand(1, 1000)
        y = np.zeros_like(x)
        mimo = MIMO()
        self.assertTrue(np.all(mimo.get_real_transferfunctions(x, y) == 0))


def test(verbosity=2):
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMIMO)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
        

if __name__ == '__main__':    
    test()
