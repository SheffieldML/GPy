import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def K_symmetric(np.ndarray[double, ndim=2] B, np.ndarray[int, ndim=1] X):
    N = X.size
    K = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            K[n,m] = B[X[n],X[m]]
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def K_asymmetric(np.ndarray[double, ndim=2] B, np.ndarray[int, ndim=1] X, np.ndarray[int, ndim=1] X2):
    N = X.size
    M = X2.size
    K = np.zeros((N, M))
    for n in range(N):
        for m in range(M):
            K[n,m] = B[X[n],X2[m]]
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def gradient_reduce(int D, np.ndarray[double, ndim=2] dL_dK, np.ndarray[int, ndim=1] index, np.ndarray[int, ndim=1] index2):
        dL_dK_small = np.zeros((D, D))
        N = index.size
        M = index2.size
        for i in range(M):
            for j in range(N):
                dL_dK_small[index[j] + D*index2[i]] += dL_dK[i+j*M];
        return dL_dK_small



