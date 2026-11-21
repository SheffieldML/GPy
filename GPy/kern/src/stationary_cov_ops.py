"""Pure NumPy stationary covariance-of-r primitives (CIP-000F Stage 2).

Covers Exponential / OU, Matern-3/2 and Matern-5/2. RBF lives in ``rbf_ops``.
"""

from __future__ import annotations

import numpy as np

from . import stationary_ops

_MAX_SCALED_R = 1e8
_SQRT3 = np.sqrt(3.0)
_SQRT5 = np.sqrt(5.0)


def _variance(variance):
    return float(np.asarray(variance).reshape(-1)[0])


def exponential_k_of_r(r, variance):
    r = np.asarray(r, dtype=np.float64)
    return _variance(variance) * np.exp(-r)


def exponential_dk_dr(r, variance):
    return -exponential_k_of_r(r, variance)


def matern32_k_of_r(r, variance):
    r = np.asarray(r, dtype=np.float64)
    return _variance(variance) * (1.0 + _SQRT3 * r) * np.exp(-_SQRT3 * r)


def matern32_dk_dr(r, variance):
    r = np.asarray(r, dtype=np.float64)
    return -3.0 * _variance(variance) * r * np.exp(-_SQRT3 * r)


def matern52_k_of_r(r, variance):
    r = np.asarray(r, dtype=np.float64)
    return _variance(variance) * (1.0 + _SQRT5 * r + 5.0 / 3.0 * r**2) * np.exp(-_SQRT5 * r)


def matern52_dk_dr(r, variance):
    r = np.asarray(r, dtype=np.float64)
    v = _variance(variance)
    return v * (10.0 / 3.0 * r - 5.0 * r - 5.0 * _SQRT5 / 3.0 * r**2) * np.exp(-_SQRT5 * r)


def K_exponential(X, variance, lengthscale, X2=None, ARD=False, max_scaled_r=_MAX_SCALED_R):
    r = stationary_ops.scaled_dist(X, lengthscale, X2=X2, ARD=ARD, max_scaled_r=max_scaled_r)
    return exponential_k_of_r(r, variance)


def K_matern32(X, variance, lengthscale, X2=None, ARD=False, max_scaled_r=_MAX_SCALED_R):
    r = stationary_ops.scaled_dist(X, lengthscale, X2=X2, ARD=ARD, max_scaled_r=max_scaled_r)
    return matern32_k_of_r(r, variance)


def K_matern52(X, variance, lengthscale, X2=None, ARD=False, max_scaled_r=_MAX_SCALED_R):
    r = stationary_ops.scaled_dist(X, lengthscale, X2=X2, ARD=ARD, max_scaled_r=max_scaled_r)
    return matern52_k_of_r(r, variance)
