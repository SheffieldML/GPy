# Copyright James Hensman and Max Zwiessele 2014, 2015
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_triangular

from . import linalg

# CIP-000E Phase 2: Cython extensions removed; flag retained for call-site guards.
use_choleskies_cython = False
choleskies_cython = None


def safe_root(N):
    i = np.sqrt(N)
    j = int(i)
    if i != j:
        raise ValueError("N is not square!")
    return j


def _flat_to_triang_pure(flat_mat):
    """Historical Python loops (slow CIP baseline)."""
    N, D = flat_mat.shape
    M = (-1 + safe_root(8 * N + 1)) // 2
    ret = np.zeros((D, M, M))
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m + 1):
                ret[d, m, mm] = flat_mat[count, d]
                count = count + 1
    return ret


@lru_cache(maxsize=128)
def _tril_lin_idx(M: int):
    """Cached lower-triangle linear indices into an ``(M, M)`` row-major matrix."""
    rows, cols = np.tril_indices(M)
    return rows, cols, (rows * M + cols).astype(np.intp)


def _flat_to_triang_idx(flat_mat):
    """CIP-000E default: cached tril scatter (matches Cython packing)."""
    flat_mat = np.asarray(flat_mat)
    N, D = flat_mat.shape
    M = (-1 + safe_root(8 * N + 1)) // 2
    rows, cols, _ = _tril_lin_idx(M)
    out = np.zeros((D, M, M), dtype=flat_mat.dtype)
    out[:, rows, cols] = flat_mat.T
    return out


def _flat_to_triang_cython(flat_mat):
    raise ImportError("choleskies_cython removed in CIP-000E Phase 2")


def _triang_to_flat_pure(L):
    """Historical Python loops (slow CIP baseline)."""
    D, _, M = L.shape

    N = M * (M + 1) // 2
    flat = np.empty((N, D))
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m + 1):
                flat[count, d] = L[d, m, mm]
                count = count + 1
    return flat


def _triang_to_flat_idx(L):
    """CIP-000E default: cached ``np.take`` gather (matches Cython packing)."""
    L = np.asarray(L)
    D, M, _ = L.shape
    _, _, lin = _tril_lin_idx(M)
    return np.take(L.reshape(D, M * M), lin, axis=1).T.copy()


def _triang_to_flat_cython(L):
    raise ImportError("choleskies_cython removed in CIP-000E Phase 2")


def _backprop_gradient_pure(dL, L):
    """
    Given the derivative of an objective fn with respect to the cholesky L,
    compute the derivate with respect to the original matrix K, defined as

        K = LL^T

    where L was obtained by Cholesky decomposition

    Slow triple-loop reference (CIP-000E oracle). Prefer :func:`cholesky_vjp`.
    """
    dL_dK = np.tril(dL).copy()
    N = L.shape[0]
    for k in range(N - 1, -1, -1):
        for j in range(k + 1, N):
            for i in range(j, N):
                dL_dK[i, k] -= dL_dK[i, j] * L[j, k]
                dL_dK[j, k] -= dL_dK[i, j] * L[i, k]
        for j in range(k + 1, N):
            dL_dK[j, k] /= L[k, k]
            dL_dK[k, k] -= L[j, k] * dL_dK[j, k]
        dL_dK[k, k] /= 2 * L[k, k]
    return dL_dK


def cholesky_vjp(dL, L):
    """Reverse-mode derivative through ``K = L @ L.T`` (pure NumPy/BLAS).

    Pure numerical primitive (CIP-000E / CIP-000F Stage 2): explicit array
    inputs and output, no ``paramz`` / model / observer coupling, no mutation
    of caller-owned buffers.

    Parameters
    ----------
    dL : array_like, shape (N, N)
        Cotangent with respect to the lower-triangular Cholesky factor ``L``.
        Only the lower triangle is used.
    L : array_like, shape (N, N)
        Lower-triangular Cholesky factor of ``K``.

    Returns
    -------
    dK : ndarray, shape (N, N)
        Cotangent with respect to ``K``, lower-triangular (same convention as
        the historical GPy Cython / pure implementations).

    Notes
    -----
    Implements the same recurrence as :func:`_backprop_gradient_pure` and the
    Cython BLAS path, but uses matrix–vector products so work is done by the
    linked BLAS without forming a dense ``Dk + Dk.T`` temporary.
    """
    d = np.tril(np.asarray(dL, dtype=np.float64)).copy()
    L = np.asarray(L, dtype=np.float64)
    if d.shape != L.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("dL and L must be square arrays of the same shape")
    N = L.shape[0]
    for k in range(N - 1, -1, -1):
        if k < N - 1:
            Dk = d[k + 1 :, k + 1 :]
            Lk = L[k + 1 :, k]
            # (Dk + Dk.T) @ Lk without allocating Dk + Dk.T (double-counts diag,
            # matching the pure triple loop / Cython dsymv + diag correction).
            tmp = Dk @ Lk
            tmp += Dk.T @ Lk
            d[k + 1 :, k] -= tmp
            d[k + 1 :, k] *= 1.0 / L[k, k]
            d[k, k] -= d[k + 1 :, k] @ Lk
        d[k, k] *= 0.5 / L[k, k]
    return d


# Historical alias used throughout GPy and tests.
_backprop_gradient_blas = cholesky_vjp


def _phi_tril_halfdiag(A):
    """Lower triangle with diagonal halved (Murray Φ)."""
    out = np.tril(A)
    diag = np.einsum("ii->i", out)
    diag *= 0.5
    return out


def _solve_tril(A, B, trans=False):
    """Solve ``tril(A) @ X = B`` (or with ``A.T`` if ``trans``). Empty-safe."""
    if B.size == 0:
        return B
    return solve_triangular(A, B, lower=True, trans=trans)


def _chol_symbolic_rev(L, Lbar):
    """Murray symbolic reverse: ``tril(dK)`` from ``tril(dL)``."""
    P = _phi_tril_halfdiag(L.T @ Lbar)
    return _phi_tril_halfdiag(_solve_tril(L, _solve_tril(L, P + P.T, trans=True).T, trans=True))


def cholesky_vjp_blocked(dL, L, block_size=128):
    """Murray blocked reverse-mode Cholesky VJP (Level-3 BLAS friendly).

    Same contract as :func:`cholesky_vjp`: ``dL, L → tril(dK)`` for ``K = L @ L.T``,
    pure arrays, no model coupling. Implements the blocked algorithm from
    Murray (2016), *Differentiation of the Cholesky decomposition*
    (arXiv:1602.07527), using triangular solves and matrix–matrix products.

    When ``N <= block_size`` the whole factor is handled by the symbolic
    reverse (most accurate). Multi-block diagonals use :func:`cholesky_vjp`
    after the ``Dbar -= tril(Cbar.T @ C)`` update — applying the symbolic
    reverse to that updated block is less stable on ill-conditioned ``K``.

    Parameters
    ----------
    dL, L : array_like, shape (N, N)
        As in :func:`cholesky_vjp`.
    block_size : int, optional
        Diagonal block width ``N_b``. Optimal value is machine- and ``N``-
        dependent; ``128`` is a reasonable default on current hosts.

    Returns
    -------
    dK : ndarray, shape (N, N)
        Lower-triangular cotangent w.r.t. ``K``.
    """
    d = np.tril(np.asarray(dL, dtype=np.float64)).copy()
    L = np.asarray(L, dtype=np.float64)
    if d.shape != L.shape or d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("dL and L must be square arrays of the same shape")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    N = L.shape[0]
    if block_size >= N:
        d[:] = _chol_symbolic_rev(L, d)
        return d
    for k in range(N, -1, -block_size):
        j = max(0, k - block_size)
        R = L[j:k, :j]
        D = L[j:k, j:k]
        B = L[k:, :j]
        C = L[k:, j:k]
        Rbar = d[j:k, :j]
        Dbar = d[j:k, j:k]
        Bbar = d[k:, :j]
        Cbar = d[k:, j:k]
        Cbar[:] = _solve_tril(D, Cbar.T, trans=True).T
        Bbar -= Cbar @ R
        Dbar[:] = np.tril(Dbar) - np.tril(Cbar.T @ C)
        # Unblocked recurrence on the updated diagonal block (CIP-000E numerics).
        Dbar[:] = cholesky_vjp(Dbar, D)
        Rbar -= Cbar.T @ B + (Dbar + Dbar.T) @ R
    return d


def triang_to_cov(L):
    return np.dstack([np.dot(L[:, :, i], L[:, :, i].T) for i in range(L.shape[-1])])


def multiple_dpotri(Ls):
    return np.array([linalg.dpotri(np.asfortranarray(Ls[i]), lower=1)[0] for i in range(Ls.shape[0])])


def indexes_to_fix_for_low_rank(rank, size):
    """
    Work out which indexes of the flatteneed array should be fixed if we want
    the cholesky to represent a low rank matrix
    """
    # first we'll work out what to keep, and the do the set difference.

    # here are the indexes of the first column, which are the triangular numbers
    n = np.arange(size)
    triangulars = (n**2 + n) / 2
    keep = []
    for i in range(rank):
        keep.append(triangulars[i:] + i)
    # add the diagonal
    keep.append(triangulars[1:] - 1)
    keep.append((size**2 + size) / 2 - 1)  # the very last element
    keep = np.hstack(keep)

    return np.setdiff1d(np.arange((size**2 + size) / 2), keep)


# CIP-000E: Murray blocked backprop + tril_indices maps are the defaults.
# Cython helpers remain importable only while extensions still ship.
backprop_gradient = cholesky_vjp_blocked
flat_to_triang = _flat_to_triang_idx
triang_to_flat = _triang_to_flat_idx
