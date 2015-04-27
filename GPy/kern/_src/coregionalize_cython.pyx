#cython: boundscheck=False
#cython: wraparound=False
import cython
import numpy as np
cimport numpy as np

def K_symmetric(np.ndarray[double, ndim=2] B, np.ndarray[int, ndim=1] X):
    cdef int N = X.size
    cdef np.ndarray[np.double_t, ndim=2] K = np.empty((N, N))
    for n in range(N):
        for m in range(N):
            K[n,m] = B[X[n],X[m]]
    return K

def K_asymmetric(np.ndarray[double, ndim=2] B, np.ndarray[int, ndim=1] X, np.ndarray[int, ndim=1] X2):
    cdef int N = X.size
    cdef int M = X2.size
    cdef np.ndarray[np.double_t, ndim=2] K = np.empty((N, M))
    for n in range(N):
        for m in range(M):
            K[n,m] = B[X[n],X2[m]]
    return K

def gradient_reduce(int D, np.ndarray[double, ndim=2] dL_dK, np.ndarray[int, ndim=1] index, np.ndarray[int, ndim=1] index2):
        cdef np.ndarray[np.double_t, ndim=2] dL_dK_small = np.zeros((D, D))
        cdef int N = index.size
        cdef int M = index2.size
        for i in range(M):
            for j in range(N):
                dL_dK_small[index[j],index2[i]] += dL_dK[i,j];
        return dL_dK_small



