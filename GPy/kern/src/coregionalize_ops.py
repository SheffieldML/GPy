"""Pure NumPy/BLAS primitives for coregionalize reductions (CIP-000E).

Array-in / array-out helpers with no ``Kern`` / ``paramz`` coupling.
"""

from __future__ import annotations

import numpy as np


def gradient_reduce(dL_dK, index, index2, output_dim):
    """Reduce a full ``dL_dK`` onto the ``output_dim × output_dim`` B-space.

    Implements the CIP-cited one-hot form matching GPy's historical layout::

        out = (O.T @ dL_dK @ O2).T

    where ``O[n, index[n]] = 1`` and ``O2[m, index2[m]] = 1``. The transpose
    matches ``_gradient_reduce_numpy`` / Cython, which store
    ``out[j, i] = sum dL_dK[n, m]`` over ``index[n] == i`` and
    ``index2[m] == j``.

    Parameters
    ----------
    dL_dK : array_like, shape (N, M)
    index : array_like, shape (N,)
        Integer output indices for rows of ``dL_dK``.
    index2 : array_like, shape (M,)
        Integer output indices for columns of ``dL_dK``.
    output_dim : int
        Number of outputs ``D``.

    Returns
    -------
    out : ndarray, shape (D, D)
    """
    dL_dK = np.asarray(dL_dK, dtype=np.float64)
    index = np.asarray(index, dtype=np.intp).ravel()
    index2 = np.asarray(index2, dtype=np.intp).ravel()
    D = int(output_dim)
    if dL_dK.ndim != 2:
        raise ValueError("dL_dK must be 2-D")
    N, M = dL_dK.shape
    if index.shape[0] != N or index2.shape[0] != M:
        raise ValueError("index / index2 length must match dL_dK shape")
    if D < 1:
        raise ValueError("output_dim must be >= 1")
    if index.min(initial=0) < 0 or index.max(initial=0) >= D:
        raise ValueError("index values out of range for output_dim")
    if index2.min(initial=0) < 0 or index2.max(initial=0) >= D:
        raise ValueError("index2 values out of range for output_dim")

    O = np.zeros((N, D), dtype=np.float64)
    O[np.arange(N), index] = 1.0
    O2 = np.zeros((M, D), dtype=np.float64)
    O2[np.arange(M), index2] = 1.0
    return (O.T @ dL_dK @ O2).T
