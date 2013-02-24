# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class KernelTests(unittest.TestCase):
    def test_kerneltie(self):
        K = GPy.kern.rbf(5, ARD=True)
        K.tie_param('[01]')
        K.constrain_fixed('2')
        X = np.random.rand(5,5)
        Y = np.ones((5,1))
        m = GPy.models.GP_regression(X,Y,K)
        print m
        self.assertTrue(m.checkgrad())



if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
