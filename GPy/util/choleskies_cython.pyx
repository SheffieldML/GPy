# Copyright James Hensman and Alan Saul 2015

import numpy as np
cimport numpy as np

from . import linalg

def flat_to_triang(np.ndarray[double, ndim=2] flat, int M):
    """take a matrix N x D and return a M X M x D array where

    N = M(M+1)/2

    the lower triangluar portion of the d'th slice of the result is filled by the d'th column of flat.
    This is the weave implementation
    """
    cdef int N = flat.shape[0]
    cdef int D = flat.shape[1]
    cdef int count = 0
    cdef np.ndarray[double, ndim=3] ret = np.zeros((M, M, D))
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m+1):
                ret[m, mm, d] = flat[count,d]
                count += 1
    return ret

def triang_to_flat(np.ndarray[double, ndim=3] L):
    cdef int M = L.shape[0]
    cdef int D = L.shape[2]
    cdef int N = M*(M+1)/2
    cdef int count = 0
    cdef np.ndarray[double, ndim=2] flat = np.empty((N, D))
    for d in range(D):
        count = 0
        for m in range(M):
            for mm in range(m+1):
                flat[count,d] = L[m, mm, d]
                count += 1
    return flat


