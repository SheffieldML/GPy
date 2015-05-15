#cython: wraparaound=False
#cython: boundscheck=False
#cython: nonecheck=False

# Copyright James Hensman and Alan Saul 2015

import numpy as np
cimport numpy as np

def flat_to_triang(np.ndarray[double, ndim=2] flat, int M):
    """take a matrix N x D and return a D X M x M array where

    N = M(M+1)/2

    the lower triangluar portion of the d'th slice of the result is filled by the d'th column of flat.
    """
    cdef int D = flat.shape[1]
    cdef int N = flat.shape[0]
    cdef int count = 0
    cdef np.ndarray[double, ndim=3] ret = np.zeros((D, M, M))
    cdef int d, m, mm
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m+1):
                ret[d, m, mm] = flat[count,d]
                count += 1
    return ret

def triang_to_flat(np.ndarray[double, ndim=3] L):
    cdef int D = L.shape[0]
    cdef int M = L.shape[1]
    cdef int N = M*(M+1)/2
    cdef int count = 0
    cdef np.ndarray[double, ndim=2] flat = np.empty((N, D))
    cdef int d, m, mm
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m+1):
                flat[count,d] = L[d, m, mm]
                count += 1
    return flat


def backprop_gradient(np.ndarray[double, ndim=2] dL, np.ndarray[double, ndim=2] L):
    cdef np.ndarray[double, ndim=2] dL_dK = np.tril(dL).copy()
    cdef int N = L.shape[0]
    cdef int k, j, i
    for k in range(N - 1, -1, -1):
        for j in range(k + 1, N):
            for i in range(j, N):
                dL_dK[i, k] -= dL_dK[i, j] * L[j, k]
                dL_dK[j, k] -= dL_dK[i, j] * L[i, k]
        for j in range(k + 1, N):
            dL_dK[j, k] /= L[k, k]
            dL_dK[k, k] -= L[j, k] * dL_dK[j, k]
        dL_dK[k, k] /= (2. * L[k, k])
    return dL_dK
