#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
import cython
import numpy as np
cimport numpy as np

def K_symmetric(np.ndarray[double, ndim=2] B, np.ndarray[np.int64_t, ndim=1] X):
    cdef int N = X.size
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] K = np.empty((N, N))
    with nogil:
        for n in range(N):
            for m in range(N):
                K[n, m] = B[X[n], X[m]]
    return K

def K_asymmetric(np.ndarray[double, ndim=2] B, np.ndarray[np.int64_t, ndim=1] X, np.ndarray[np.int64_t, ndim=1] X2):
    cdef int N = X.size
    cdef int M = X2.size
    cdef np.ndarray[np.double_t, ndim=2, mode='c'] K = np.empty((N, M))
    with nogil:
        for n in range(N):
            for m in range(M):
                K[n, m] = B[X[n], X2[m]]
    return K

def gradient_reduce(int D, np.ndarray[double, ndim=2] dL_dK, np.ndarray[np.int64_t, ndim=1] index, np.ndarray[np.int64_t, ndim=1] index2):
        cdef np.ndarray[np.double_t, ndim=2, mode='c'] dL_dK_small = np.zeros((D, D))
        cdef int N = index.size
        cdef int M = index2.size
        with nogil:
            for i in range(N):
                for j in range(M):
                    dL_dK_small[index2[j],index[i]] += dL_dK[i,j];
        return dL_dK_small



