#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

ctypedef np.float64_t DTYPE_t
 
cdef extern from "stationary_utils.h":
    void _grad_X "_grad_X" (int N, int D, int M, double* X, double* X2, double* tmp, double* grad) nogil

cdef extern from "stationary_utils.h":
    void _lengthscale_grads "_lengthscale_grads" (int N, int M, int Q, double* tmp, double* X, double* X2, double* grad) nogil
 
def grad_X(int N, int D, int M,
        np.ndarray[DTYPE_t, ndim=2] _X,
        np.ndarray[DTYPE_t, ndim=2] _X2,
        np.ndarray[DTYPE_t, ndim=2] _tmp,
        np.ndarray[DTYPE_t, ndim=2] _grad):
    cdef double *X = <double*> _X.data
    cdef double *X2 = <double*> _X2.data
    cdef double *tmp = <double*> _tmp.data
    cdef double *grad = <double*> _grad.data
    with nogil:
        _grad_X(N, D, M, X, X2, tmp, grad) # return nothing, work in place.

@cython.cdivision(True)
def grad_X_cython(int N, int D, int M, double[:,:] X, double[:,:] X2, double[:,:] tmp, double[:,:] grad):
    cdef int n,d,nd,m
    for nd in prange(N * D, nogil=True):
        n = nd / D
        d = nd % D
        grad[n,d] = 0.0
        for m in range(M):
            grad[n,d] += tmp[n, m] * (X[n, d] - X2[m, d])

def lengthscale_grads_in_c(int N, int M, int Q,
        np.ndarray[DTYPE_t, ndim=2] _tmp,
        np.ndarray[DTYPE_t, ndim=2] _X,
        np.ndarray[DTYPE_t, ndim=2] _X2,
        np.ndarray[DTYPE_t, ndim=1] _grad):
    cdef double *tmp = <double*> _tmp.data
    cdef double *X = <double*> _X.data
    cdef double *X2 = <double*> _X2.data
    cdef double *grad = <double*> _grad.data
    with nogil:
        _lengthscale_grads(N, M, Q, tmp, X, X2, grad) # return nothing, work in place.

def lengthscale_grads(int N, int M, int Q, double[:,:] tmp, double[:,:] X, double[:,:] X2, double[:] grad):
    cdef int q, n, m
    cdef double gradq, dist
    with nogil:
        for q in range(Q):
            grad[q] = 0.0
            for n in range(N):
                for m in range(M):
                    dist = X[n,q] - X2[m,q]
                    grad[q] += tmp[n, m] * dist * dist
