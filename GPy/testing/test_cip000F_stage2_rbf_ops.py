"""Golden-value tests for CIP-000F Stage 2 RBF pure numerical primitives.

Exercises ``rbf_ops`` / ``stationary_ops`` without constructing a GPy model or
``paramz`` ``Param``, and checks parity against ``GPy.kern.RBF`` wrappers.
"""

from __future__ import annotations

import numpy as np

import GPy
from GPy.kern.src import rbf_ops, stationary_ops


def _rng():
    return np.random.RandomState(0)


class TestRBFOpsPure:
    def test_k_of_r_and_derivatives_finite(self):
        r = np.array([0.0, 0.5, 2.0, 50.0])
        k = rbf_ops.k_of_r(r, 1.7)
        assert np.all(np.isfinite(k))
        np.testing.assert_allclose(k[0], 1.7)
        np.testing.assert_allclose(k[-1], 0.0, atol=1e-12)
        dk = rbf_ops.dk_dr(r, 1.7)
        d2 = rbf_ops.dk2_drdr(r, 1.7)
        assert np.all(np.isfinite(dk))
        assert np.all(np.isfinite(d2))
        np.testing.assert_allclose(dk[0], 0.0)

    def test_K_and_kdiag_without_param_objects(self):
        rng = _rng()
        X = rng.randn(8, 2)
        X2 = rng.randn(5, 2)
        variance = 0.8
        lengthscale = np.array([1.2, 0.7])
        K = rbf_ops.K(X, variance, lengthscale, X2=X2, ARD=True)
        assert K.shape == (8, 5)
        assert np.all(np.isfinite(K))
        kd = rbf_ops.kdiag(X.shape[0], variance)
        np.testing.assert_allclose(kd, variance)

    def test_scaled_dist_matches_kernel_wrapper(self):
        rng = _rng()
        X = rng.randn(6, 1)
        kern = GPy.kern.RBF(1, variance=1.3, lengthscale=0.9)
        r_ops = stationary_ops.scaled_dist(X, 0.9)
        r_kern = kern._scaled_dist(X)
        np.testing.assert_allclose(r_ops, r_kern)

    def test_wrapper_parity_full_and_diag(self):
        rng = _rng()
        X = rng.randn(10, 2)
        X2 = rng.randn(7, 2)
        kern = GPy.kern.RBF(2, variance=1.1, lengthscale=[0.8, 1.4], ARD=True)
        variance = np.asarray(kern.variance, dtype=float).reshape(-1)[0]
        lengthscale = np.asarray(kern.lengthscale, dtype=float)
        K_pure = rbf_ops.K(X, variance, lengthscale, X2=X2, ARD=True)
        np.testing.assert_allclose(K_pure, kern.K(X, X2))
        np.testing.assert_allclose(rbf_ops.kdiag(X.shape[0], variance), kern.Kdiag(X))

        r = kern._scaled_dist(X, X2)
        np.testing.assert_allclose(rbf_ops.k_of_r(r, variance), kern.K_of_r(r))
        np.testing.assert_allclose(rbf_ops.dk_dr(r, variance), kern.dK_dr(r))
