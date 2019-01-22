#cython: wraparaound=False
#cython: boundscheck=False
#cython: nonecheck=False

# Copyright James Hensman and Alan Saul 2015

import numpy as np
from cython.parallel import prange, parallel
cimport numpy as np
cimport scipy.linalg.cython_blas as cblas

np.import_array()

def flat_to_triang(double[:, :] flat, int M):
    """take a matrix N x D and return a D X M x M array where

    N = M(M+1)/2

    the lower triangluar portion of the d'th slice of the result is filled by the d'th column of flat.
    """
    cdef int D = flat.shape[1]
    cdef int N = flat.shape[0]
    cdef int count = 0
    cdef double[:, :, ::1] ret = np.zeros((D, M, M))
    cdef int d, m, mm
    with nogil:
        for d in range(D):
            count = 0
            for m in range(M):
                for mm in range(m+1):
                    ret[d, m, mm] = flat[count,d]
                    count += 1
    return ret

def triang_to_flat(double[:, :, :] L):
    cdef int D = L.shape[0]
    cdef int M = L.shape[1]
    cdef int N = M*(M+1)/2
    cdef int count = 0
    cdef double[:, ::1] flat = np.empty((N, D))
    cdef int d, m, mm
    with nogil:
        for d in range(D):
            count = 0
            for m in range(M):
                for mm in range(m+1):
                    flat[count,d] = L[d, m, mm]
                    count += 1
    return flat

def backprop_gradient(double[:, :] dL, double[:, :] L):
    cdef double[:, ::1] dL_dK = np.tril(dL)
    cdef int N = L.shape[0]
    cdef int k, j, i
    with nogil:
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

def backprop_gradient_par(double[:,:] dL, double[:,:] L):
    cdef double[:,::1] dL_dK = np.tril(dL)
    cdef int N = L.shape[0]
    cdef int k, j, i
    with nogil:
        for k in range(N - 1, -1, -1):
            with parallel():
                for i in prange(k + 1, N):
                    for j in range(k+1, i+1):
                        dL_dK[i, k] -= dL_dK[i, j] * L[j, k]
                    for j in range(i, N):
                        dL_dK[i, k] -= dL_dK[j, i] * L[j, k]
            for j in range(k + 1, N):
                dL_dK[j, k] /= L[k, k]
                dL_dK[k, k] -= L[j, k] * dL_dK[j, k]
            dL_dK[k, k] /= (2. * L[k, k])
    return dL_dK

cdef void chol_backprop(int N, double[:, ::1] dL, double[:, ::1] L) nogil:
    cdef int i, k, n

    # DSYMV required constant arguments
    cdef double alpha=-1, beta=1
    cdef int incx=N

    # DSCAL required arguments
    cdef double scale

    dL[N - 1, N - 1] /= (2. * L[N - 1, N - 1])
    for k in range(N-2, -1, -1):
        n = N-k-1
        cblas.dsymv(uplo='u', n=&n, alpha=&alpha, a=&dL[k + 1, k + 1], lda=&N, x=&L[k + 1, k], incx=&incx,
                    beta=&beta, y=&dL[k + 1, k], incy=&N)

        for i in xrange(0, N - k - 1):
            dL[k + 1 + i, k] -= dL[k + i+ 1, k + i + 1] * L[k + 1 + i, k]

        scale = 1.0 / L[k, k]
        cblas.dscal(&n, &scale , &dL[k + 1, k], &N)
#
        dL[k, k] -= cblas.ddot(&n, &dL[k + 1, k], &N, &L[k+1, k], &incx)
        dL[k, k] /= (2.0 * L[k, k])

def backprop_gradient_par_c(double[:, :] dL, double[:, :] L):
    cdef double[:, ::1] dL_dK = np.tril(dL) # makes a copy, c-contig
    cdef double[:, ::1] L_cont = np.ascontiguousarray(L)
    cdef int N = L.shape[0]
    with nogil:
        chol_backprop(N, dL_dK, L_cont)
    return np.asarray(dL_dK)
