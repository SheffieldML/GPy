"""Pure NumPy RBF / squared-exponential primitives (CIP-000F Stage 2).

Array-in / array-out helpers with no ``Kern`` / ``paramz`` coupling. Wrappers
on ``GPy.kern.RBF`` unwrap parameter values once and delegate here.
"""

from __future__ import annotations

import numpy as np

from . import stationary_ops

# Match ``Stationary._MAX_SCALED_R`` / RBF overflow clip used in ``K_of_r``.
_MAX_SCALED_R = 1e8
_R2_EXP_CLIP = 746.0


def k_of_r(r, variance):
    """RBF covariance from scaled distances: ``variance * exp(-0.5 r^2)``."""
    r = np.asarray(r, dtype=np.float64)
    variance = float(np.asarray(variance).reshape(-1)[0])
    with np.errstate(over="ignore", invalid="ignore"):
        r2 = np.square(r)
    r2 = np.minimum(np.nan_to_num(r2, nan=np.inf, posinf=np.inf), _R2_EXP_CLIP)
    return variance * np.exp(-0.5 * r2)


def dk_dr(r, variance):
    """First derivative of the RBF covariance wrt scaled distance ``r``."""
    r = np.asarray(r, dtype=np.float64)
    return -r * k_of_r(r, variance)


def dk2_drdr(r, variance):
    """Second derivative of the RBF covariance wrt scaled distance ``r``."""
    r = np.asarray(r, dtype=np.float64)
    return (r**2 - 1.0) * k_of_r(r, variance)


def kdiag(n, variance):
    """Diagonal of an RBF (or any stationary) Gram matrix: constant variance."""
    n = int(n)
    variance = float(np.asarray(variance).reshape(-1)[0])
    out = np.empty(n, dtype=np.float64)
    out[:] = variance
    return out


def K(X, variance, lengthscale, X2=None, ARD=False, max_scaled_r=_MAX_SCALED_R):
    """Full RBF Gram matrix from plain arrays (no ``Param`` objects)."""
    r = stationary_ops.scaled_dist(X, lengthscale, X2=X2, ARD=ARD, max_scaled_r=max_scaled_r)
    return k_of_r(r, variance)
