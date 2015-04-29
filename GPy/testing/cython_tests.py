import numpy as np
import scipy as sp
from GPy.util import choleskies
import GPy

"""
These tests make sure that the opure python and cython codes work the same
"""

class CythonTestChols(np.testing.TestCase):
    def setUp(self):
        self.flat = np.random.randn(45, 5)
        self.triang = np.dstack([np.eye(20)[:,:,None] for i in range(3)])
    def test_flat_to_triang(self):
        L1 = choleskies._flat_to_triang_pure(self.flat)
        L2 = choleskies._flat_to_triang_cython(self.flat)
        np.testing.assert_allclose(L1, L2)
    def test_triang_to_flat(self):
        A1 = choleskies._triang_to_flat_pure(self.triang)
        A2 = choleskies._triang_to_flat_cython(self.triang)
        np.testing.assert_allclose(A1, A2)

class test_stationary(np.testing.TestCase):
    def setUp(self):
        self.k = GPy.kern.RBF(10)
        self.X = np.random.randn(300,10)
        self.Z = np.random.randn(20,10)
        self.dKxx = np.random.randn(300,300)
        self.dKzz = np.random.randn(20,20)
        self.dKxz = np.random.randn(300,20)

    def test_square_gradX(self):
        g1 = self.k._gradients_X_cython(self.dKxx, self.X)
        g2 = self.k._gradients_X_pure(self.dKxx, self.X)
        np.testing.assert_allclose(g1, g2)

    def test_rect_gradx(self):
        g1 = self.k._gradients_X_cython(self.dKxz, self.X, self.Z)
        g2 = self.k._gradients_X_pure(self.dKxz, self.X, self.Z)
        np.testing.assert_allclose(g1, g2)

    def test_square_lengthscales(self):
        g1 = self.k._lengthscale_grads_pure(self.dKxx, self.X, self.X)
        g2 = self.k._lengthscale_grads_cython(self.dKxx, self.X, self.X)
        np.testing.assert_allclose(g1, g2)

    def test_rect_lengthscales(self):
        g1 = self.k._lengthscale_grads_pure(self.dKxz, self.X, self.Z)
        g2 = self.k._lengthscale_grads_cython(self.dKxz, self.X, self.Z)
        np.testing.assert_allclose(g1, g2)

class test_choleskies_backprop(np.testing.TestCase):
    def setUp(self):
        self.dL, self.L = np.random.randn(2, 100, 100)
    def test(self):
        r1 = GPy.util.choleskies._backprop_gradient_pure(self.dL, self.L)
        r2 = GPy.util.choleskies.choleskies_cython.backprop_gradient(self.dL, self.L)
        np.testing.assert_allclose(r1, r2)







