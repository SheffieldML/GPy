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
    from ..kern.src import stationary_cython as stationary_cython

    stationary_cython_working = config.getboolean("cython", "working")
except ImportError:
    stationary_cython_working = False

"""
These tests make sure that the pure python and cython codes work the same
"""


class CythonTestChols:
    def setup_method(self):
        self.flat = np.random.randn(45, 5)
        self.triang = np.array([np.eye(20) for i in range(3)])

    @pytest.mark.skipif(
        not choleskies_cython_working,
        "Cython cholesky module has not been built on this machine",
    )
    def test_flat_to_triang(self):
        L1 = choleskies._flat_to_triang_pure(self.flat)
        L2 = choleskies._flat_to_triang_cython(self.flat)
        assert np.allclose(L1, L2), "Triang mismatch!"

    @pytest.mark.skipif(
        not choleskies_cython_working,
        "Cython cholesky module has not been built on this machine",
    )
    def test_triang_to_flat(self):
        A1 = choleskies._triang_to_flat_pure(self.triang)
        A2 = choleskies._triang_to_flat_cython(self.triang)
        assert np.allclose(A1, A2), "Flat mismatch!"


class TestStationary:
    def setup_method(self):
        self.k = GPy.kern.RBF(10)
        self.X = np.random.randn(300, 10)
        self.Z = np.random.randn(20, 10)
        self.dKxx = np.random.randn(300, 300)
        self.dKzz = np.random.randn(20, 20)
        self.dKxz = np.random.randn(300, 20)

    def test_square_gradX_blas_matches_pure(self):
        """CIP-000E: GEMM gradients_X matches the loop oracle."""
        self.setup_method()
        g_pure = self.k._gradients_X_pure(self.dKxx, self.X)
        g_blas = self.k._gradients_X_blas(self.dKxx, self.X)
        assert np.allclose(g_pure, g_blas), "BLAS/pure square gradX mismatch!"

    def test_rect_gradX_blas_matches_pure(self):
        self.setup_method()
        g_pure = self.k._gradients_X_pure(self.dKxz, self.X, self.Z)
        g_blas = self.k._gradients_X_blas(self.dKxz, self.X, self.Z)
        assert np.allclose(g_pure, g_blas), "BLAS/pure rect gradX mismatch!"

    def test_square_lengthscales_blas_matches_pure(self):
        """CIP-000E: matmul lengthscale grads match the loop oracle."""
        self.setup_method()
        g_pure = self.k._lengthscale_grads_pure(self.dKxx, self.X, self.X)
        g_blas = self.k._lengthscale_grads_blas(self.dKxx, self.X, self.X)
        assert np.allclose(g_pure, g_blas), "BLAS/pure square lengthscale mismatch!"

    def test_rect_lengthscales_blas_matches_pure(self):
        self.setup_method()
        g_pure = self.k._lengthscale_grads_pure(self.dKxz, self.X, self.Z)
        g_blas = self.k._lengthscale_grads_blas(self.dKxz, self.X, self.Z)
        assert np.allclose(g_pure, g_blas), "BLAS/pure rect lengthscale mismatch!"

    def test_stationary_ops_no_kern_required(self):
        """Primitives are callable without a Kern / paramz object."""
        from GPy.kern.src import stationary_ops

        rng = np.random.default_rng(0)
        tmp = rng.normal(size=(8, 5))
        X = rng.normal(size=(8, 3))
        X2 = rng.normal(size=(5, 3))
        g = stationary_ops.lengthscale_grads_sqdiff(tmp, X, X2)
        assert g.shape == (3,)
        gx = stationary_ops.gradients_X_from_tmp(tmp, X, X2)
        assert gx.shape == (8, 3)

    @pytest.mark.skipif(
        not stationary_cython_working,
        reason="Cython stationary module has not been built on this machine",
    )
    def test_square_gradX(self):
        self.setup_method()
        g1 = self.k._gradients_X_cython(self.dKxx, self.X)
        g2 = self.k._gradients_X_pure(self.dKxx, self.X)
        g3 = self.k._gradients_X_blas(self.dKxx, self.X)
        assert np.allclose(g1, g2), "Gradient mismatch on square X!"
        assert np.allclose(g1, g3), "BLAS/Cython square gradX mismatch!"

    @pytest.mark.skipif(
        not stationary_cython_working,
        reason="Cython stationary module has not been built on this machine",
    )
    def test_rect_gradx(self):
        self.setup_method()
        g1 = self.k._gradients_X_cython(self.dKxz, self.X, self.Z)
        g2 = self.k._gradients_X_pure(self.dKxz, self.X, self.Z)
        g3 = self.k._gradients_X_blas(self.dKxz, self.X, self.Z)
        assert np.allclose(g1, g2), "Gradient mismatch on rect X!"
        assert np.allclose(g1, g3), "BLAS/Cython rect gradX mismatch!"

    @pytest.mark.skipif(
        not stationary_cython_working,
        reason="Cython stationary module has not been built on this machine",
    )
    def test_square_lengthscales(self):
        self.setup_method()
        g1 = self.k._lengthscale_grads_pure(self.dKxx, self.X, self.X)
        g2 = self.k._lengthscale_grads_cython(self.dKxx, self.X, self.X)
        g3 = self.k._lengthscale_grads_blas(self.dKxx, self.X, self.X)
        assert np.allclose(g1, g2), "Gradient mismatch on square lengthscale!"
        assert np.allclose(g1, g3), "BLAS/Cython square lengthscale mismatch!"

    @pytest.mark.skipif(
        not stationary_cython_working,
        reason="Cython stationary module has not been built on this machine",
    )
    def test_rect_lengthscales(self):
        self.setup_method()
        g1 = self.k._lengthscale_grads_pure(self.dKxz, self.X, self.Z)
        g2 = self.k._lengthscale_grads_cython(self.dKxz, self.X, self.Z)
        g3 = self.k._lengthscale_grads_blas(self.dKxz, self.X, self.Z)
        assert np.allclose(g1, g2), "Gradient mismatch on rect lengthscale!"
        assert np.allclose(g1, g3), "BLAS/Cython rect lengthscale mismatch!"


class TestCholeskiesBackprop:
    def setup_method(self):
        a = np.random.randn(10, 12)
        A = a.dot(a.T)
        self.L = GPy.util.linalg.jitchol(A)
        self.dL = np.random.randn(10, 10)

    def test_backprop_blas_matches_pure(self):
        """CIP-000E: NumPy/BLAS cholesky_vjp matches the triple-loop oracle."""
        self.setup_method()
        r_pure = choleskies._backprop_gradient_pure(self.dL, self.L)
        r_blas = choleskies.cholesky_vjp(self.dL, self.L)
        assert np.allclose(r_pure, r_blas), "BLAS/pure gradient mismatch!"

    def test_backprop_murray_blocked_matches_pure(self):
        """CIP-000E: Murray blocked VJP matches the triple-loop oracle."""
        self.setup_method()
        r_pure = choleskies._backprop_gradient_pure(self.dL, self.L)
        r_blocked = choleskies.cholesky_vjp_blocked(self.dL, self.L, block_size=4)
        assert np.allclose(r_pure, r_blocked), "Murray blocked/pure gradient mismatch!"

    def test_cholesky_vjp_no_model_required(self):
        """Primitive is callable without a GPy model or paramz object."""
        L = np.tril(np.eye(4) + 0.1 * np.arange(16).reshape(4, 4))
        # Ensure SPD Cholesky factor
        K = L @ L.T + np.eye(4)
        L = np.linalg.cholesky(K)
        dL = np.random.randn(4, 4)
        dK = choleskies.cholesky_vjp(dL, L)
        assert dK.shape == (4, 4)
        assert np.allclose(dK, np.tril(dK))
        dK_b = choleskies.cholesky_vjp_blocked(dL, L, block_size=2)
        assert np.allclose(dK, dK_b)

    @pytest.mark.skipif(
        not choleskies_cython_working,
        reason="Cython cholesky module has not been built on this machine",
    )
    def test_backprop(self):
        self.setup_method()
        r1 = choleskies._backprop_gradient_pure(self.dL, self.L)
        r2 = choleskies_cython.backprop_gradient(self.dL, self.L)
        r3 = choleskies_cython.backprop_gradient_par_c(self.dL, self.L)
        r4 = choleskies.cholesky_vjp(self.dL, self.L)
        r5 = choleskies.cholesky_vjp_blocked(self.dL, self.L, block_size=4)
        assert np.allclose(r1, r2), "Gradient mismatch!"
        assert np.allclose(r1, r3), "Gradient mismatch!"
        assert np.allclose(r1, r4), "BLAS/Cython gradient mismatch!"
        assert np.allclose(r1, r5), "Murray blocked/Cython gradient mismatch!"


def test_state_space_cython_available_when_built():
    """CIP-000E Phase 0: relative import must expose cython_code_available."""
    try:
        from GPy.models import state_space_cython as _ssc  # noqa: F401
    except ImportError:
        pytest.skip("state_space_cython extension has not been built")

    # Fresh attribute check via package path (same as installed use).
    from GPy.models import state_space_main as ssm

    assert ssm.cython_code_available is True
