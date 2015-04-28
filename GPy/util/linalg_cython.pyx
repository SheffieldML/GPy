cimport numpy as np
from cpython cimport bool
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def symmetrify(np.ndarray[double, ndim=2] A, bool upper):
    cdef int N = A.shape[0]
    if not upper:
        for i in xrange(N):
            for j in xrange(i):
                A[j, i] = A[i, j]
    else:
        for j in xrange(N):
            for i in xrange(j):
                A[j, i] = A[i, j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cholupdate(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=2] L, int N):
    cdef double r
    cdef double c
    cdef double s
    for j in xrange(N):
        r = np.sqrt(L[j,j]*L[j,j] + x[j]*x[j])
        c = r / L[j,j]
        s = x[j] / L[j,j]
        L[j,j] = r
        for i in xrange(j):
            L[i,j] = (L[i,j] + s*x[i])/c
            x[i] = c*x[i] - s*L[i,j];
        r = np.sqrt(L[j,j])
