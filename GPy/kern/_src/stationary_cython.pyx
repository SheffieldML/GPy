#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
 
cdef extern from "stationary_utils.h":
    void _grad_X "_grad_X" (int N, int D, int M, double* X, double* X2, double* tmp, double* grad)
 
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
