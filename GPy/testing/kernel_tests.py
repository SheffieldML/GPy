# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy

verbose = False

try:
    import sympy
    SYMPY_AVAILABLE=True
except ImportError:
    SYMPY_AVAILABLE=False


class KernelTests(unittest.TestCase):
    def test_kerneltie(self):
        K = GPy.kern.rbf(5, ARD=True)
        K.tie_params('.*[01]')
        K.constrain_fixed('2')
        X = np.random.rand(5,5)
        Y = np.ones((5,1))
        m = GPy.models.GPRegression(X,Y,K)
        self.assertTrue(m.checkgrad())

    def test_rbfkernel(self):
        kern = GPy.kern.rbf(5)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_rbf_sympykernel(self):
        if SYMPY_AVAILABLE:
            kern = GPy.kern.rbf_sympy(5)
            self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_eq_sympykernel(self):
        if SYMPY_AVAILABLE:
            kern = GPy.kern.eq_sympy(5, 3)
            self.assertTrue(GPy.kern.kern_test(kern, output_ind=4, verbose=verbose))

    def test_ode1_eqkernel(self):
        if SYMPY_AVAILABLE:
            kern = GPy.kern.ode1_eq(3)
            self.assertTrue(GPy.kern.kern_test(kern, output_ind=1, verbose=verbose, X_positive=True))

    def test_rbf_invkernel(self):
        kern = GPy.kern.rbf_inv(5)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_Matern32kernel(self):
        kern = GPy.kern.Matern32(5)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_Matern52kernel(self):
        kern = GPy.kern.Matern52(5)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_linearkernel(self):
        kern = GPy.kern.linear(5)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_periodic_exponentialkernel(self):
        kern = GPy.kern.periodic_exponential(1)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_periodic_Matern32kernel(self):
        kern = GPy.kern.periodic_Matern32(1)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_periodic_Matern52kernel(self):
        kern = GPy.kern.periodic_Matern52(1)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_rational_quadratickernel(self):
        kern = GPy.kern.rational_quadratic(1)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_gibbskernel(self):
        kern = GPy.kern.gibbs(5, mapping=GPy.mappings.Linear(5, 1))
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_heterokernel(self):
        kern = GPy.kern.hetero(5, mapping=GPy.mappings.Linear(5, 1), transform=GPy.core.transformations.logexp())
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_mlpkernel(self):
        kern = GPy.kern.mlp(5)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_polykernel(self):
        kern = GPy.kern.poly(5, degree=4)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    def test_fixedkernel(self):
        """
        Fixed effect kernel test
        """
        X = np.random.rand(30, 4)
        K = np.dot(X, X.T)
        kernel = GPy.kern.fixed(4, K)
        kern = GPy.kern.poly(5, degree=4)
        self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))

    # def test_coregionalization(self):
    #     X1 = np.random.rand(50,1)*8
    #     X2 = np.random.rand(30,1)*5
    #     index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
    #     X = np.hstack((np.vstack((X1,X2)),index))
    #     Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
    #     Y2 = np.sin(X2) + np.random.randn(*X2.shape)*0.05 + 2.
    #     Y = np.vstack((Y1,Y2))

    #     k1 = GPy.kern.rbf(1) + GPy.kern.bias(1)
    #     k2 = GPy.kern.coregionalize(2,1)
    #     kern = k1**k2
    #     self.assertTrue(GPy.kern.kern_test(kern, verbose=verbose))


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
