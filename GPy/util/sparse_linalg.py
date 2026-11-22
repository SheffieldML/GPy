# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Shared sparse inducing-side factorisation helpers (CIP-0013).

Tokens match GPc / GPc CIP-0016: ``lq`` | ``stable`` | ``legacy``.
LQ factorisation follows Seeger et al. (2017) / GPc CIP-001B.
"""

from __future__ import annotations

import numpy as np


_CANONICAL = frozenset({"lq", "stable", "legacy"})
_ALIASES = {
    "stablefactor": "stable",
    "chol": "stable",
    "gram": "legacy",
}


def parse_sparse_linalg(mode: str) -> str:
    """Return canonical ``lq`` | ``stable`` | ``legacy``.

    Case-insensitive. Read aliases: ``stablefactor``/``chol`` → ``stable``,
    ``gram`` → ``legacy``.
    """
    if not isinstance(mode, str):
        raise TypeError(f"sparse_linalg must be a string, got {type(mode)!r}")
    key = mode.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _CANONICAL:
        raise ValueError(f"sparse_linalg={mode!r} is not supported. Use 'lq', 'stable', or 'legacy'.")
    return key


def lq_factor(W: np.ndarray, normalize_signs: bool = True):
    """Factor wide ``W`` (m×n, n≥m) as ``W = L @ Q``.

    Uses QR of ``W.T`` (NumPy has no ``gelqf``). Rows of ``Q`` are orthonormal.
    When ``normalize_signs`` is True (default), flip rows so ``diag(L) > 0``.
    """
    W = np.ascontiguousarray(W, dtype=float)
    m, n = W.shape
    if n < m:
        raise ValueError(f"lq_factor expects wide matrix (n>=m); got shape {W.shape}")
    # W.T = Qr @ Rr  ⇒  W = Rr.T @ Qr.T
    Qr, Rr = np.linalg.qr(W.T, mode="reduced")
    L = np.ascontiguousarray(Rr.T)
    Q = np.ascontiguousarray(Qr.T)
    if normalize_signs:
        s = np.sign(np.diag(L))
        s[s == 0.0] = 1.0
        L = L * s[:, None]
        Q = s[:, None] * Q
    return L, Q


def lq_adjoint(L: np.ndarray, Q: np.ndarray, Lbar: np.ndarray, Qbar: np.ndarray):
    """Seeger §4.3 reverse-mode: cotangents of ``L,Q`` → cotangent of ``W=LQ``.

    Parameters
    ----------
    L : (m, m) lower triangular
    Q : (m, n) with orthonormal rows
    Lbar, Qbar : same shapes as L, Q

    Returns
    -------
    Wbar : (m, n)
    """
    L = np.ascontiguousarray(L, dtype=float)
    Q = np.ascontiguousarray(Q, dtype=float)
    Lbar = np.ascontiguousarray(Lbar, dtype=float)
    Qbar = np.ascontiguousarray(Qbar, dtype=float)
    # M = L.T @ Lbar - Qbar @ Q.T
    M = L.T @ Lbar - Qbar @ Q.T
    # copyltu(M): keep lower incl. diag, mirror to upper
    Msym = np.tril(M)
    Msym = Msym + np.tril(M, -1).T
    # S = Qbar + copyltu(M) @ Q
    S = Qbar + Msym @ Q
    # Wbar = L^{-T} S
    Wbar = np.linalg.solve(L.T, S)
    return Wbar


def factor_I_plus_AA(features: np.ndarray, mode: str):
    """Factor ``I + F F.T`` for ``F`` shape (m, n).

    Returns
    -------
    L : (m, m) lower factor (Cholesky or LQ ``L``)
    A : (m, m) Gram ``F @ F.T`` (kept for callers that need ``trace(A)``)
    Q : (m, m+n) or None — LQ orthogonal factor when ``mode=='lq'``
    """
    mode = parse_sparse_linalg(mode)
    F = np.ascontiguousarray(features, dtype=float)
    m = F.shape[0]
    A = F @ F.T
    if mode == "stable":
        from .linalg import jitchol

        L = jitchol(np.eye(m) + A)
        return L, A, None
    if mode == "lq":
        W = np.hstack([np.eye(m), F])
        L, Q = lq_factor(W)
        return L, A, Q
    raise NotImplementedError(
        "sparse_linalg='legacy' is not implemented in GPy yet (CIP-0013 Phase B). Use 'lq' or 'stable'."
    )
