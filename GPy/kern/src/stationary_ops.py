"""Pure NumPy/BLAS primitives for stationary kernel derivatives (CIP-000E).

These are array-in / array-out helpers with no ``Kern`` / ``paramz`` coupling.
They implement the CIP-cited winning forms (not the slow per-dimension Python
loops used as historical fallbacks).
"""

from __future__ import annotations

import numpy as np


def lengthscale_grads_sqdiff(tmp, X, X2):
    """Sum ``tmp[n,m] * (X[n,q] - X2[m,q])**2`` for each input dimension ``q``.

    CIP-000E form (one matmul + reductions)::

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
    row = tmp.sum(axis=1)
    col = tmp.sum(axis=0)
    return (X * X).T @ row + (X2 * X2).T @ col - 2.0 * (X * (tmp @ X2)).sum(axis=0)


def gradients_X_from_tmp(tmp, X, X2):
    """Sum ``tmp[n,m] * (X[n,q] - X2[m,q])`` over ``m`` for each ``n,q``.

    CIP-000E GEMM form::

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
    return X * tmp.sum(axis=1)[:, None] - tmp @ X2
