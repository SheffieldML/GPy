import numpy as np
from GPy.util import choleskies
import GPy
import pytest

from ..util.config import config

try:
    from ..util import choleskies_cython

    choleskies_cython_working = config.getboolean("cython", "working")
except ImportError:
    choleskies_cython_working = False

try:
    from ..kern.src import stationary_cython

    stationary_cython_working = config.getboolean("cython", "working")
except ImportError:
    stationary_cython_working = False

"""
These tests make sure that the pure python and cython codes work the same
"""


@pytest.mark.skipif(
    not choleskies_cython_working,
    "Cython cholesky module has not been built on this machine",
)
class CythonTestChols:
    def setup(self):
        self.flat = np.random.randn(45, 5)
        self.triang = np.array([np.eye(20) for i in range(3)])

    def test_flat_to_triang(self):
        L1 = choleskies._flat_to_triang_pure(self.flat)
        L2 = choleskies._flat_to_triang_cython(self.flat)
        assert np.allclose(L1, L2), "Triang mismatch!"

    def test_triang_to_flat(self):
        A1 = choleskies._triang_to_flat_pure(self.triang)
        A2 = choleskies._triang_to_flat_cython(self.triang)
        assert np.allclose(A1, A2), "Flat mismatch!"


@pytest.mark.skipif(
    not stationary_cython_working,
    "Cython stationary module has not been built on this machine",
)
class TestStationary:
    def setup(self):
        self.k = GPy.kern.RBF(10)
        self.X = np.random.randn(300, 10)
        self.Z = np.random.randn(20, 10)
        self.dKxx = np.random.randn(300, 300)
        self.dKzz = np.random.randn(20, 20)
        self.dKxz = np.random.randn(300, 20)

    def test_square_gradX(self):
        self.setup()
        g1 = self.k._gradients_X_cython(self.dKxx, self.X)
        g2 = self.k._gradients_X_pure(self.dKxx, self.X)
        assert np.allclose(g1, g2), "Gradient mismatch on square X!"

    def test_rect_gradx(self):
        self.setup()
        g1 = self.k._gradients_X_cython(self.dKxz, self.X, self.Z)
        g2 = self.k._gradients_X_pure(self.dKxz, self.X, self.Z)
        assert np.allclose(g1, g2), "Gradient mismatch on rect X!"

    def test_square_lengthscales(self):
        self.setup()
        g1 = self.k._lengthscale_grads_pure(self.dKxx, self.X, self.X)
        g2 = self.k._lengthscale_grads_cython(self.dKxx, self.X, self.X)
        assert np.allclose(g1, g2), "Gradient mismatch on square lengthscale!"

    def test_rect_lengthscales(self):
        self.setup()
        g1 = self.k._lengthscale_grads_pure(self.dKxz, self.X, self.Z)
        g2 = self.k._lengthscale_grads_cython(self.dKxz, self.X, self.Z)
        assert np.allclose(g1, g2), "Gradient mismatch on rect lengthscale!"


@pytest.mark.skipif(
    not choleskies_cython_working,
    "Cython cholesky module has not been built on this machine",
)
class TestCholeskiesBackprop:
    def setup(self):
        a = np.random.randn(10, 12)
        A = a.dot(a.T)
        self.L = GPy.util.linalg.jitchol(A)
        self.dL = np.random.randn(10, 10)

    def test_backprop(self):
        self.setup()
        r1 = choleskies._backprop_gradient_pure(self.dL, self.L)
        r2 = choleskies_cython.backprop_gradient(self.dL, self.L)
        r3 = choleskies_cython.backprop_gradient_par_c(self.dL, self.L)
        assert np.allclose(r1, r2), "Gradient mismatch!"
        assert np.allclose(r1, r3), "Gradient mismatch!"
