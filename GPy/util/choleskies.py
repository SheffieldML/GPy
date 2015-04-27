# Copyright James Hensman and Max Zwiessele 2014
# Licensed under the GNU GPL version 3.0

import numpy as np
from . import linalg
from .config import config

import choleskies_cython

def safe_root(N):
    i = np.sqrt(N)
    j = int(i)
    if i != j:
        raise ValueError("N is not square!")
    return j

def _flat_to_triang_pure(flat_mat):
    N, D = flat_mat.shape
    M = (-1 + safe_root(8*N+1))//2
    ret = np.zeros((M, M, D))
    count = 0
    for m in range(M):
        for mm in range(m+1):
            for d in range(D):
              ret.flat[d + m*D*M + mm*D] = flat_mat.flat[count];
              count = count+1
    return ret

def _flat_to_triang_cython(flat_mat):
    N, D = flat_mat.shape
    M = (-1 + safe_root(8*N+1))//2
    return choleskies_cython.flat_to_triang(flat_mat, M)


def _triang_to_flat_pure(L):
    M, _, D = L.shape

    N = M*(M+1)//2
    flat = np.empty((N, D))
    count = 0;
    for m in range(M):
        for mm in range(m+1):
            for d in range(D):
                flat.flat[count] = L.flat[d + m*D*M + mm*D];
                count = count +1
    return flat

def _triang_to_flat_cython(L):
    return choleskies_cython.triang_to_flat(L)

def triang_to_cov(L):
    return np.dstack([np.dot(L[:,:,i], L[:,:,i].T) for i in range(L.shape[-1])])

def multiple_dpotri(Ls):
    return np.dstack([linalg.dpotri(np.asfortranarray(Ls[:,:,i]), lower=1)[0] for i in range(Ls.shape[-1])])

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
else:
    triang_to_flat =  _triang_to_flat_pure
    flat_to_triang = _flat_to_triang_pure

