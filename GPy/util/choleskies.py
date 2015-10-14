# Copyright James Hensman and Max Zwiessele 2014, 2015
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from . import linalg
from .config import config
try:
    from . import choleskies_cython
    config.set('cython', 'working', 'True')
except ImportError:
    config.set('cython', 'working', 'False')

def safe_root(N):
    i = np.sqrt(N)
    j = int(i)
    if i != j:
        raise ValueError("N is not square!")
    return j

def _flat_to_triang_pure(flat_mat):
    N, D = flat_mat.shape
    M = (-1 + safe_root(8*N+1))//2
    ret = np.zeros((D, M, M))
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m+1):
                ret[d,m, mm] = flat_mat[count, d];
                count = count+1
    return ret

def _flat_to_triang_cython(flat_mat):
    N, D = flat_mat.shape
    M = (-1 + safe_root(8*N+1))//2
    return choleskies_cython.flat_to_triang(flat_mat, M)


def _triang_to_flat_pure(L):
    D, _, M = L.shape

    N = M*(M+1)//2
    flat = np.empty((N, D))
    for d in range(D):
        count = 0;
        for m in range(M):
            for mm in range(m+1):
                flat[count,d] = L[d, m, mm]
                count = count +1
    return flat

def _triang_to_flat_cython(L):
    return choleskies_cython.triang_to_flat(L)

def _backprop_gradient_pure(dL, L):
    """
    Given the derivative of an objective fn with respect to the cholesky L,
    compute the derivate with respect to the original matrix K, defined as

        K = LL^T

    where L was obtained by Cholesky decomposition
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
        dL_dK[k, k] /= (2 * L[k, k])
    return dL_dK

def triang_to_cov(L):
    return np.dstack([np.dot(L[:,:,i], L[:,:,i].T) for i in range(L.shape[-1])])

def multiple_dpotri(Ls):
    return np.array([linalg.dpotri(np.asfortranarray(Ls[i]), lower=1)[0] for i in range(Ls.shape[0])])

def indexes_to_fix_for_low_rank(rank, size):
    """
    Work out which indexes of the flatteneed array should be fixed if we want
    the cholesky to represent a low rank matrix
    """
    #first we'll work out what to keep, and the do the set difference.

    #here are the indexes of the first column, which are the triangular numbers
    n = np.arange(size)
    triangulars = (n**2 + n) / 2
    keep = []
    for i in range(rank):
        keep.append(triangulars[i:] + i)
    #add the diagonal
    keep.append(triangulars[1:]-1)
    keep.append((size**2 + size)/2 -1)# the very last element
    keep = np.hstack(keep)

    return np.setdiff1d(np.arange((size**2+size)/2), keep)


if config.getboolean('cython', 'working'):
    triang_to_flat = _triang_to_flat_cython
    flat_to_triang = _flat_to_triang_cython
    backprop_gradient = choleskies_cython.backprop_gradient_par_c
else:
    backprop_gradient = _backprop_gradient_pure
    triang_to_flat =  _triang_to_flat_pure
    flat_to_triang = _flat_to_triang_pure
