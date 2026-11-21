"""Pure NumPy/BLAS primitives for stationary kernel derivatives (CIP-000E).

These are array-in / array-out helpers with no ``Kern`` / ``paramz`` coupling.
They implement the CIP-cited winning forms (not the slow per-dimension Python
loops used as historical fallbacks).

Numerical note
---------------
The expanded identities ``(x-y)^2 = x^2+y^2-2xy`` and
``Σ tmp·(x-y) = X·rowsum - tmp@X2`` lose precision when ``X`` and ``X2`` share
a large common offset (e.g. Unix timestamps). Cython takes the difference
first and does not suffer this. We subtract a shared column mean before the
expanded forms; that is translation-invariant for both identities and restores
accuracy when the offset is shared.
"""

from __future__ import annotations

import numpy as np


def _center_pair(X, X2):
    """Subtract a shared column mean so expanded (x−y) forms stay accurate."""
    c = X.mean(axis=0)
    return X - c, X2 - c


def lengthscale_grads_sqdiff(tmp, X, X2):
    """Sum ``tmp[n,m] * (X[n,q] - X2[m,q])**2`` for each input dimension ``q``.

    CIP-000E form (one matmul + reductions), after shared mean-centring::

        row = tmp.sum(1); col = tmp.sum(0)
        (X**2).T @ row + (X2**2).T @ col - 2 * (X * (tmp @ X2)).sum(0)

    Parameters
    ----------
    tmp : array_like, shape (N, M)
        Weights (typically ``dK_dr * dL_dK / r``-style factors already formed
        by the kernel wrapper).
    X : array_like, shape (N, Q)
    X2 : array_like, shape (M, Q)

    Returns
    -------
    grads : ndarray, shape (Q,)
        Per-dimension squared-difference reductions (before ``/ lengthscale**3``
        scaling applied by the kernel wrapper).
    """
    tmp = np.asarray(tmp, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    if tmp.ndim != 2 or X.ndim != 2 or X2.ndim != 2:
        raise ValueError("tmp, X, X2 must be 2-D")
    if tmp.shape[0] != X.shape[0] or tmp.shape[1] != X2.shape[0] or X.shape[1] != X2.shape[1]:
        raise ValueError("incompatible shapes for tmp, X, X2")
    X, X2 = _center_pair(X, X2)
    row = tmp.sum(axis=1)
    col = tmp.sum(axis=0)
    return (X * X).T @ row + (X2 * X2).T @ col - 2.0 * (X * (tmp @ X2)).sum(axis=0)


def gradients_X_from_tmp(tmp, X, X2):
    """Sum ``tmp[n,m] * (X[n,q] - X2[m,q])`` over ``m`` for each ``n,q``.

    CIP-000E GEMM form, after shared mean-centring::

        X * tmp.sum(1)[:, None] - tmp @ X2

    Parameters
    ----------
    tmp : array_like, shape (N, M)
        Pre-multiplied ``inv_dist * dK_dr * dL_dK`` (and symmetrised when
        ``X2 is X``), as formed by the kernel wrapper.
    X : array_like, shape (N, Q)
    X2 : array_like, shape (M, Q)

    Returns
    -------
    grad : ndarray, shape (N, Q)
        Unscaled input gradients (wrapper divides by ``lengthscale**2``).
    """
    tmp = np.asarray(tmp, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    if tmp.ndim != 2 or X.ndim != 2 or X2.ndim != 2:
        raise ValueError("tmp, X, X2 must be 2-D")
    if tmp.shape[0] != X.shape[0] or tmp.shape[1] != X2.shape[0] or X.shape[1] != X2.shape[1]:
        raise ValueError("incompatible shapes for tmp, X, X2")
    X, X2 = _center_pair(X, X2)
    return X * tmp.sum(axis=1)[:, None] - tmp @ X2


def unscaled_dist(X, X2=None):
    """Euclidean distances between rows of ``X`` and ``X2`` (CIP-000F Stage 2).

    Direct differences keep translation invariance under large shared offsets
    and give exact ``r=0`` for coincident points.
    """
    X = np.asarray(X, dtype=np.float64)
    if X2 is None:
        X2 = X
        r2 = np.sum((X[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(r2, 0.0)
        r2 = np.clip(r2, 0.0, np.inf)
        return np.sqrt(r2)
    X2 = np.asarray(X2, dtype=np.float64)
    r2 = np.sum((X[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
    r2 = np.clip(r2, 0.0, np.inf)
    return np.sqrt(r2)


def scaled_dist(X, lengthscale, X2=None, ARD=False, max_scaled_r=1e8):
    """Lengthscale-scaled distances used by stationary kernels."""
    X = np.asarray(X, dtype=np.float64)
    lengthscale = np.asarray(lengthscale, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if ARD:
            X_sc = X / lengthscale
            X2_sc = None if X2 is None else np.asarray(X2, dtype=np.float64) / lengthscale
            r = unscaled_dist(X_sc, X2_sc)
        else:
            r = unscaled_dist(X, X2) / lengthscale
    r = np.nan_to_num(r, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return np.minimum(r, max_scaled_r)
