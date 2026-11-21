"""CIP-000E: NumPy/BLAS path parity (Cython extensions removed in Phase 2)."""

import numpy as np
from GPy.util import choleskies, linalg
import GPy
import pytest


class TestCholsMaps:
    def setup_method(self):
        self.flat = np.random.randn(45, 5)
        self.triang = np.array([np.eye(20) for i in range(3)])

    def test_flat_to_triang_idx_matches_pure(self):
        L1 = choleskies._flat_to_triang_pure(self.flat)
        L2 = choleskies._flat_to_triang_idx(self.flat)
        assert np.allclose(L1, L2), "Triang mismatch!"
        assert np.allclose(choleskies.flat_to_triang(self.flat), L2)

    def test_triang_to_flat_idx_matches_pure(self):
        A1 = choleskies._triang_to_flat_pure(self.triang)
        A2 = choleskies._triang_to_flat_idx(self.triang)
        assert np.allclose(A1, A2), "Flat mismatch!"
        assert np.allclose(choleskies.triang_to_flat(self.triang), A2)

    def test_roundtrip(self):
        L = choleskies.flat_to_triang(self.flat)
        flat2 = choleskies.triang_to_flat(L)
        assert np.allclose(self.flat, flat2)

    def test_removed_cython_map_stubs_raise(self):
        with pytest.raises(ImportError, match="Phase 2"):
            choleskies._flat_to_triang_cython(self.flat)
        with pytest.raises(ImportError, match="Phase 2"):
            choleskies._triang_to_flat_cython(self.triang)


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

    def test_stationary_ops_stable_under_large_shared_offset(self):
        """Expanded BLAS forms must match direct (x-y) loops after mean-shift.

        Without centring, offsets like Unix timestamps (~1e9) destroy accuracy
        of ``x^2+y^2-2xy`` / ``X·rowsum - tmp@X2`` (CIP-000E follow-up).
        """
        from GPy.kern.src import stationary_ops

        rng = np.random.default_rng(0)
        N, M, Q = 40, 25, 3
        tmp = rng.normal(size=(N, M))
        X0 = rng.normal(size=(N, Q))
        X20 = rng.normal(size=(M, Q))
        offset = 1e9
        X, X2 = X0 + offset, X20 + offset

        def loop_len():
            return np.array([np.sum(tmp * np.square(X[:, q : q + 1] - X2[:, q : q + 1].T)) for q in range(Q)])

        def loop_gx():
            g = np.empty((N, Q))
            for q in range(Q):
                g[:, q] = np.sum(tmp * (X[:, q][:, None] - X2[:, q][None, :]), axis=1)
            return g

        assert np.allclose(stationary_ops.lengthscale_grads_sqdiff(tmp, X, X2), loop_len(), rtol=0, atol=1e-8)
        assert np.allclose(stationary_ops.gradients_X_from_tmp(tmp, X, X2), loop_gx(), rtol=0, atol=1e-8)


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
        K = L @ L.T + np.eye(4)
        L = np.linalg.cholesky(K)
        dL = np.random.randn(4, 4)
        dK = choleskies.cholesky_vjp(dL, L)
        assert dK.shape == (4, 4)
        assert np.allclose(dK, np.tril(dK))
        dK_b = choleskies.cholesky_vjp_blocked(dL, L, block_size=2)
        assert np.allclose(dK, dK_b)

    def test_default_backprop_is_murray_blocked(self):
        assert choleskies.backprop_gradient is choleskies.cholesky_vjp_blocked

    def test_cholesky_vjp_rejects_bad_shapes_and_block_size(self):
        L = np.eye(3)
        dL = np.eye(3)
        with pytest.raises(ValueError, match="same shape"):
            choleskies.cholesky_vjp(np.eye(2), L)
        with pytest.raises(ValueError, match="same shape"):
            choleskies.cholesky_vjp_blocked(np.eye(2), L)
        with pytest.raises(ValueError, match="block_size"):
            choleskies.cholesky_vjp_blocked(dL, L, block_size=0)

    def test_blocked_matches_pure_on_illconditioned_rbf(self):
        """Multi-block Murray path must track the pure recurrence on stiff K.

        Well-conditioned random SPD tests miss the updated-diagonal symbolic
        instability; RBF + tiny jitter exposes it (CIP-000E follow-up).
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 2))
        d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
        K = np.exp(-0.5 * d2) + 1e-8 * np.eye(80)
        L = np.linalg.cholesky(K)
        dL = rng.normal(size=(80, 80))
        pure = choleskies._backprop_gradient_pure(dL, L)
        blocked = choleskies.cholesky_vjp_blocked(dL, L, block_size=16)
        assert np.allclose(pure, blocked, rtol=0, atol=1e-5)


class TestSymmetrify:
    def test_blocked_matches_numpy_lower(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(64, 64))
        B = A.copy()
        C = A.copy()
        linalg._symmetrify_numpy(B, upper=False)
        linalg._symmetrify_blocked(C, upper=False, block_size=16)
        assert np.allclose(B, C)
        D = A.copy()
        linalg.symmetrify(D, upper=False)
        assert np.allclose(B, D)

    def test_blocked_matches_numpy_upper(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(64, 64))
        B = A.copy()
        C = A.copy()
        linalg._symmetrify_numpy(B, upper=True)
        linalg._symmetrify_blocked(C, upper=True, block_size=16)
        assert np.allclose(B, C)

    def test_removed_cython_symmetrify_stub_raises(self):
        with pytest.raises(ImportError, match="Phase 2"):
            linalg._symmetrify_cython(np.eye(3))

    def test_blocked_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            linalg._symmetrify_blocked(np.zeros((3, 2)))


class TestCoregionalizeReduce:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.D = 4
        self.N = 80
        self.index = rng.integers(0, self.D, size=(self.N, 1))
        self.index2 = rng.integers(0, self.D, size=(self.N, 1))
        self.dL = rng.normal(size=(self.N, self.N))
        self.k = GPy.kern.Coregionalize(1, output_dim=self.D)

    def test_blas_matches_numpy_loops(self):
        """CIP-000E: one-hot reduce matches nested-loop oracle."""
        self.setup_method()
        a = self.k._gradient_reduce_numpy(self.dL, self.index, self.index2)
        b = self.k._gradient_reduce_blas(self.dL, self.index, self.index2)
        assert np.allclose(a, b), "BLAS/loop gradient_reduce mismatch!"

    def test_ops_no_kern_required(self):
        from GPy.kern.src import coregionalize_ops

        out = coregionalize_ops.gradient_reduce(self.dL, self.index[:, 0], self.index2[:, 0], self.D)
        assert out.shape == (self.D, self.D)

    def test_ops_reject_bad_inputs(self):
        from GPy.kern.src import coregionalize_ops

        with pytest.raises(ValueError, match="2-D"):
            coregionalize_ops.gradient_reduce(np.ones(3), self.index[:, 0], self.index2[:, 0], self.D)
        with pytest.raises(ValueError, match="length"):
            coregionalize_ops.gradient_reduce(self.dL, self.index[:2, 0], self.index2[:, 0], self.D)
        with pytest.raises(ValueError, match="output_dim"):
            coregionalize_ops.gradient_reduce(self.dL, self.index[:, 0], self.index2[:, 0], 0)


def test_stationary_ops_reject_bad_shapes():
    from GPy.kern.src import stationary_ops

    tmp = np.ones((4, 3))
    X = np.ones((4, 2))
    X2 = np.ones((3, 2))
    with pytest.raises(ValueError, match="2-D"):
        stationary_ops.lengthscale_grads_sqdiff(np.ones(3), X, X2)
    with pytest.raises(ValueError, match="incompatible"):
        stationary_ops.gradients_X_from_tmp(tmp, X, np.ones((2, 2)))


def test_state_space_cython_removed():
    """CIP-000E Phase 2: Cython state-space path is gone; Python remains."""
    from GPy.models import state_space_main as ssm

    assert ssm.cython_code_available is False
    assert ssm.use_cython is False
    with pytest.raises(ImportError):
        from GPy.models import state_space_cython  # noqa: F401
