# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

class KernelTests(unittest.TestCase):
    def test_kerneltie(self):
        K = GPy.kern.rbf(5, ARD=True)
        K.tie_params('.*[01]')
        K.constrain_fixed('2')
        X = np.random.rand(5,5)
        Y = np.ones((5,1))
        m = GPy.models.GPRegression(X,Y,K)
        self.assertTrue(m.checkgrad())

    def test_fixedkernel(self):
        """
        Fixed effect kernel test
        """
        X = np.random.rand(30, 4)
        K = np.dot(X, X.T)
        kernel = GPy.kern.Fixed(4, K)
        Y = np.ones((30,1))
        m = GPy.models.GPRegression(X,Y,kernel=kernel)
        self.assertTrue(m.checkgrad())

    def test_coregionalisation(self):
        X1 = np.random.rand(50,1)*8
        X2 = np.random.rand(30,1)*5
        index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
        X = np.hstack((np.vstack((X1,X2)),index))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
        Y2 = np.sin(X2) + np.random.randn(*X2.shape)*0.05 + 2.
        Y = np.vstack((Y1,Y2))

        k1 = GPy.kern.rbf(1) + GPy.kern.bias(1)
        k2 = GPy.kern.Coregionalise(2,1)
        k = k1.prod(k2,tensor=True)
        m = GPy.models.GPRegression(X,Y,kernel=k)
        self.assertTrue(m.checkgrad())




if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
