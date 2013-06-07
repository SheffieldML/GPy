# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# tdot function courtesy of Ian Murray:
# Iain Murray, April 2013. iain contactable via iainmurray.net
# http://homepages.inf.ed.ac.uk/imurray2/code/tdot/tdot.py

import numpy as np
from scipy import linalg, weave
import types
import ctypes
from ctypes import byref, c_char, c_int, c_double # TODO
# import scipy.lib.lapack
import scipy

if np.all(np.float64((scipy.__version__).split('.')[:2]) >= np.array([0, 12])):
    import scipy.linalg.lapack as lapack
else:
    from scipy.linalg.lapack import flapack as lapack

try:
    _blaslib = ctypes.cdll.LoadLibrary(np.core._dotblas.__file__) # @UndefinedVariable
    _blas_available = True
    assert hasattr('dsyrk_',_blaslib)
    assert hasattr('dsyr_',_blaslib)
except:
    _blas_available = False

def dtrtrs(A, B, lower=0, trans=0, unitdiag=0):
    """Wrapper for lapack dtrtrs function

    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:
    """
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)

def dpotrs(A, B, lower=0):
    """Wrapper for lapack dpotrs function

    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:
    """
    return lapack.dpotrs(A, B, lower=lower)

def dpotri(A, lower=0):
    """Wrapper for lapack dpotri function

    :param A: Matrix A
    :param lower: is matrix lower (true) or upper (false)
    :returns:
    """
    return lapack.dpotri(A, lower=lower)

def trace_dot(a, b):
    """
    efficiently compute the trace of the matrix product of a and b
    """
    return np.sum(a * b)

def mdot(*args):
    """Multiply all the arguments using matrix product rules.
    The output is equivalent to multiplying the arguments one by one
    from left to right using dot().
    Precedence can be controlled by creating tuples of arguments,
    for instance mdot(a,((b,c),d)) multiplies a (a*((b*c)*d)).
    Note that this means the output of dot(a,b) and mdot(a,b) will differ if
    a or b is a pure tuple of numbers.
    """
    if len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return _mdot_r(args[0], args[1])
    else:
        return _mdot_r(args[:-1], args[-1])

def _mdot_r(a, b):
    """Recursive helper for mdot"""
    if type(a) == types.TupleType:
        if len(a) > 1:
            a = mdot(*a)
        else:
            a = a[0]
    if type(b) == types.TupleType:
        if len(b) > 1:
            b = mdot(*b)
        else:
            b = b[0]
    return np.dot(a, b)

def jitchol(A, maxtries=5):
    A = np.asfortranarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError, "not pd: non-positive diagonal elements"
        jitter = diagA.mean() * 1e-6
        while maxtries > 0 and np.isfinite(jitter):
            print 'Warning: adding jitter of {:.10e}'.format(jitter)
            try:
                return linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)
            except:
                jitter *= 10
            finally:
                maxtries -= 1
        raise linalg.LinAlgError, "not positive definite, even with jitter."



def jitchol_old(A, maxtries=5):
    """
    :param A : An almost pd square matrix

    :rval L: the Cholesky decomposition of A

    .. Note:
      Adds jitter to K, to enforce positive-definiteness
      if stuff breaks, please check:
      np.allclose(sp.linalg.cholesky(XXT, lower = True), np.triu(sp.linalg.cho_factor(XXT)[0]).T)
    """
    try:
        return linalg.cholesky(A, lower=True)
    except linalg.LinAlgError:
        diagA = np.diag(A)
        if np.any(diagA < 0.):
            raise linalg.LinAlgError, "not pd: negative diagonal elements"
        jitter = diagA.mean() * 1e-6
        for i in range(1, maxtries + 1):
            print '\rWarning: adding jitter of {:.10e}                        '.format(jitter),
            try:
                return linalg.cholesky(A + np.eye(A.shape[0]).T * jitter, lower=True)
            except:
                jitter *= 10

        raise linalg.LinAlgError, "not positive definite, even with jitter."

def pdinv(A, *args):
    """
    :param A: A DxD pd numpy array

    :rval Ai: the inverse of A
    :rtype Ai: np.ndarray
    :rval L: the Cholesky decomposition of A
    :rtype L: np.ndarray
    :rval Li: the Cholesky decomposition of Ai
    :rtype Li: np.ndarray
    :rval logdet: the log of the determinant of A
    :rtype logdet: float64
    """
    L = jitchol(A, *args)
    logdet = 2.*np.sum(np.log(np.diag(L)))
    Li = chol_inv(L)
    Ai, _ = lapack.dpotri(L)
    # Ai = np.tril(Ai) + np.tril(Ai,-1).T
    symmetrify(Ai)

    return Ai, L, Li, logdet


def chol_inv(L):
    """
    Inverts a Cholesky lower triangular matrix

    :param L: lower triangular matrix
    :rtype: inverse of L

    """

    return lapack.dtrtri(L, lower=True)[0]


def multiple_pdinv(A):
    """
    Arguments
    ---------
    :param A: A DxDxN numpy array (each A[:,:,i] is pd)

    Returns
    -------
    invs : the inverses of A
    hld: 0.5* the log of the determinants of A
    """
    N = A.shape[-1]
    chols = [jitchol(A[:, :, i]) for i in range(N)]
    halflogdets = [np.sum(np.log(np.diag(L[0]))) for L in chols]
    invs = [lapack.dpotri(L[0], True)[0] for L in chols]
    invs = [np.triu(I) + np.triu(I, 1).T for I in invs]
    return np.dstack(invs), np.array(halflogdets)


def PCA(Y, input_dim):
    """
    Principal component analysis: maximum likelihood solution by SVD

    Arguments
    ---------
    :param Y: NxD np.array of data
    :param input_dim: int, dimension of projection

    Returns
    -------
    :rval X: - Nxinput_dim np.array of dimensionality reduced data
    W - input_dimxD mapping from X to Y
    """
    if not np.allclose(Y.mean(axis=0), 0.0):
        print "Y is not zero mean, centering it locally (GPy.util.linalg.PCA)"

        # Y -= Y.mean(axis=0)

    Z = linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    [X, W] = [Z[0][:, 0:input_dim], np.dot(np.diag(Z[1]), Z[2]).T[:, 0:input_dim]]
    v = X.std(axis=0)
    X /= v;
    W *= v;
    return X, W.T


def tdot_numpy(mat, out=None):
    return np.dot(mat, mat.T, out)

def tdot_blas(mat, out=None):
    """returns np.dot(mat, mat.T), but faster for large 2D arrays of doubles."""
    if (mat.dtype != 'float64') or (len(mat.shape) != 2):
        return np.dot(mat, mat.T)
    nn = mat.shape[0]
    if out is None:
        out = np.zeros((nn, nn))
    else:
        assert(out.dtype == 'float64')
        assert(out.shape == (nn, nn))
        # FIXME: should allow non-contiguous out, and copy output into it:
        assert(8 in out.strides)
        # zeroing needed because of dumb way I copy across triangular answer
        out[:] = 0.0

    # # Call to DSYRK from BLAS
    # If already in Fortran order (rare), and has the right sorts of strides I
    # could avoid the copy. I also thought swapping to cblas API would allow use
    # of C order. However, I tried that and had errors with large matrices:
    # http://homepages.inf.ed.ac.uk/imurray2/code/tdot/tdot_broken.py
    mat = np.asfortranarray(mat)
    TRANS = c_char('n')
    N = c_int(mat.shape[0])
    K = c_int(mat.shape[1])
    LDA = c_int(mat.shape[0])
    UPLO = c_char('l')
    ALPHA = c_double(1.0)
    A = mat.ctypes.data_as(ctypes.c_void_p)
    BETA = c_double(0.0)
    C = out.ctypes.data_as(ctypes.c_void_p)
    LDC = c_int(np.max(out.strides) / 8)
    _blaslib.dsyrk_(byref(UPLO), byref(TRANS), byref(N), byref(K),
            byref(ALPHA), A, byref(LDA), byref(BETA), C, byref(LDC))

    symmetrify(out, upper=True)

    return out

def tdot(*args, **kwargs):
    if _blas_available:
        return tdot_blas(*args, **kwargs)
    else:
        return tdot_numpy(*args, **kwargs)

def DSYR_blas(A, x, alpha=1.):
    """
    Performs a symmetric rank-1 update operation:
    A <- A + alpha * np.dot(x,x.T)

    Arguments
    ---------
    :param A: Symmetric NxN np.array
    :param x: Nx1 np.array
    :param alpha: scalar
    """
    N = c_int(A.shape[0])
    LDA = c_int(A.shape[0])
    UPLO = c_char('l')
    ALPHA = c_double(alpha)
    A_ = A.ctypes.data_as(ctypes.c_void_p)
    x_ = x.ctypes.data_as(ctypes.c_void_p)
    INCX = c_int(1)
    _blaslib.dsyr_(byref(UPLO), byref(N), byref(ALPHA),
            x_, byref(INCX), A_, byref(LDA))
    symmetrify(A, upper=True)

def DSYR_numpy(A, x, alpha=1.):
    """
    Performs a symmetric rank-1 update operation:
    A <- A + alpha * np.dot(x,x.T)

    Arguments
    ---------
    :param A: Symmetric NxN np.array
    :param x: Nx1 np.array
    :param alpha: scalar
    """
    A += alpha * np.dot(x[:, None], x[None, :])


def DSYR(*args, **kwargs):
    if _blas_available:
        return DSYR_blas(*args, **kwargs)
    else:
        return DSYR_numpy(*args, **kwargs)

def symmetrify(A, upper=False):
    """
    Take the square matrix A and make it symmetrical by copting elements from the lower half to the upper

    works IN PLACE.
    """
    N, M = A.shape
    assert N == M
    c_contig_code = """
    int iN;
    for (int i=1; i<N; i++){
      iN = i*N;
      for (int j=0; j<i; j++){
        A[i+j*N] = A[iN+j];
      }
    }
    """
    f_contig_code = """
    int iN;
    for (int i=1; i<N; i++){
      iN = i*N;
      for (int j=0; j<i; j++){
        A[iN+j] = A[i+j*N];
      }
    }
    """
    if A.flags['C_CONTIGUOUS'] and upper:
        weave.inline(f_contig_code, ['A', 'N'], extra_compile_args=['-O3'])
    elif A.flags['C_CONTIGUOUS'] and not upper:
        weave.inline(c_contig_code, ['A', 'N'], extra_compile_args=['-O3'])
    elif A.flags['F_CONTIGUOUS'] and upper:
        weave.inline(c_contig_code, ['A', 'N'], extra_compile_args=['-O3'])
    elif A.flags['F_CONTIGUOUS'] and not upper:
        weave.inline(f_contig_code, ['A', 'N'], extra_compile_args=['-O3'])
    else:
        if upper:
            tmp = np.tril(A.T)
        else:
            tmp = np.tril(A)
        A[:] = 0.0
        A += tmp
        A += np.tril(tmp, -1).T


def symmetrify_murray(A):
    A += A.T
    nn = A.shape[0]
    A[[range(nn), range(nn)]] /= 2.0

def cholupdate(L, x):
    """
    update the LOWER cholesky factor of a pd matrix IN PLACE

    if L is the lower chol. of K, then this function computes L_
    where L_ is the lower chol of K + x*x^T
    """
    support_code = """
    #include <math.h>
    """
    code = """
    double r,c,s;
    int j,i;
    for(j=0; j<N; j++){
      r = sqrt(L(j,j)*L(j,j) + x(j)*x(j));
      c = r / L(j,j);
      s = x(j) / L(j,j);
      L(j,j) = r;
      for (i=j+1; i<N; i++){
        L(i,j) = (L(i,j) + s*x(i))/c;
        x(i) = c*x(i) - s*L(i,j);
      }
    }
    """
    x = x.copy()
    N = x.size
    weave.inline(code, support_code=support_code, arg_names=['N', 'L', 'x'], type_converters=weave.converters.blitz)

def backsub_both_sides(L, X, transpose='left'):
    """ Return L^-T * X * L^-1, assumuing X is symmetrical and L is lower cholesky"""
    if transpose == 'left':
        tmp, _ = lapack.dtrtrs(L, np.asfortranarray(X), lower=1, trans=1)
        return lapack.dtrtrs(L, np.asfortranarray(tmp.T), lower=1, trans=1)[0].T
    else:
        tmp, _ = lapack.dtrtrs(L, np.asfortranarray(X), lower=1, trans=0)
        return lapack.dtrtrs(L, np.asfortranarray(tmp.T), lower=1, trans=0)[0].T

