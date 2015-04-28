#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
 
cdef extern from "stationary_utils.h":
    void _grad_X "_grad_X" (int N, int D, int M, double* X, double* X2, double* tmp, double* grad)

cdef extern from "stationary_utils.h":
    void _lengthscale_grads "_lengthscale_grads" (int N, int M, int Q, double* tmp, double* X, double* X2, double* grad)
 
def grad_X(int N, int D, int M,
        np.ndarray[DTYPE_t, ndim=2] _X,
        np.ndarray[DTYPE_t, ndim=2] _X2,
        np.ndarray[DTYPE_t, ndim=2] _tmp,
        np.ndarray[DTYPE_t, ndim=2] _grad):
    cdef double *X = <double*> _X.data
    cdef double *X2 = <double*> _X2.data
    cdef double *tmp = <double*> _tmp.data
    cdef double *grad = <double*> _grad.data
    _grad_X(N, D, M, X, X2, tmp, grad) # return nothing, work in place.

def lengthscale_grads_c(int N, int M, int Q,
        np.ndarray[DTYPE_t, ndim=2] _tmp,
        np.ndarray[DTYPE_t, ndim=2] _X,
        np.ndarray[DTYPE_t, ndim=2] _X2,
        np.ndarray[DTYPE_t, ndim=1] _grad):
    cdef double *tmp = <double*> _tmp.data
    cdef double *X = <double*> _X.data
    cdef double *X2 = <double*> _X2.data
    cdef double *grad = <double*> _grad.data
    _lengthscale_grads(N, M, Q, tmp, X, X2, grad) # return nothing, work in place.

def lengthscale_grads(int N, int M, int Q,
        np.ndarray[DTYPE_t, ndim=2] tmp,
        np.ndarray[DTYPE_t, ndim=2] X,
        np.ndarray[DTYPE_t, ndim=2] X2,
        np.ndarray[DTYPE_t, ndim=1] grad):
    for q in range(Q):
        for i in range(N):
            for j in range(M):
                grad[q] += tmp[i,j]*(X[i,q]-X2[j,q])**2

