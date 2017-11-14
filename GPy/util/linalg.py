# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# tdot function courtesy of Ian Murray:
# Iain Murray, April 2013. iain contactable via iainmurray.net
# http://homepages.inf.ed.ac.uk/imurray2/code/tdot/tdot.py

import numpy as np
from scipy import linalg
from scipy.linalg import lapack, blas
from .config import config
import logging

if config.getboolean('cython', 'working'):
    from . import linalg_cython

def force_F_ordered_symmetric(A):
    """
    return a F ordered version of A, assuming A is symmetric
    """
    if A.flags['F_CONTIGUOUS']:
        return A
    if A.flags['C_CONTIGUOUS']:
        return A.T
    else:
        return np.asfortranarray(A)

def force_F_ordered(A):
    """
    return a F ordered version of A, assuming A is triangular
    """
    if A.flags['F_CONTIGUOUS']:
        return A
    print("why are your arrays not F order?")
    return np.asfortranarray(A)

# def jitchol(A, maxtries=5):
#     A = force_F_ordered_symmetric(A)
#     L, info = lapack.dpotrf(A, lower=1)
#     if info == 0:
#         return L
#     else:
#         if maxtries==0:
#             raise linalg.LinAlgError, "not positive definite, even with jitter."
#         diagA = np.diag(A)
#         if np.any(diagA <= 0.):
#             raise linalg.LinAlgError, "not pd: non-positive diagonal elements"
#         jitter = diagA.mean() * 1e-6

#         return jitchol(A+np.eye(A.shape[0])*jitter, maxtries-1)


def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")
    import traceback
    try: raise
    except:
        logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
            '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
    return L

# def dtrtri(L, lower=1):
#     """
#     Wrapper for lapack dtrtri function
#     Inverse of L
#
#     :param L: Triangular Matrix L
#     :param lower: is matrix lower (true) or upper (false)
#     :returns: Li, info
#     """
#     L = force_F_ordered(L)
#     return lapack.dtrtri(L, lower=lower)

def dtrtrs(A, B, lower=1, trans=0, unitdiag=0):
    """
    Wrapper for lapack dtrtrs function

    DTRTRS solves a triangular system of the form

        A * X = B  or  A**T * X = B,

    where A is a triangular matrix of order N, and B is an N-by-NRHS
    matrix.  A check is made to verify that A is nonsingular.

    :param A: Matrix A(triangular)
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns: Solution to A * X = B or A**T * X = B

    """
    A = np.asfortranarray(A)
    #Note: B does not seem to need to be F ordered!
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)

def dpotrs(A, B, lower=1):
    """
    Wrapper for lapack dpotrs function
    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:
    """
    A = force_F_ordered(A)
    return lapack.dpotrs(A, B, lower=lower)

def dpotri(A, lower=1):
    """
    Wrapper for lapack dpotri function

    DPOTRI - compute the inverse of a real symmetric positive
      definite matrix A using the Cholesky factorization A =
      U**T*U or A = L*L**T computed by DPOTRF

    :param A: Matrix A
    :param lower: is matrix lower (true) or upper (false)
    :returns: A inverse

    """

    A = force_F_ordered(A)
    R, info = lapack.dpotri(A, lower=lower) #needs to be zero here, seems to be a scipy bug

    symmetrify(R)
    return R, info

def pddet(A):
    """
    Determinant of a positive definite matrix, only symmetric matricies though
    """
    L = jitchol(A)
    logdetA = 2*sum(np.log(np.diag(L)))
    return logdetA

def trace_dot(a, b):
    """
    Efficiently compute the trace of the matrix product of a and b
    """
    return np.einsum('ij,ji->', a, b)

def mdot(*args):
    """
    Multiply all the arguments using matrix product rules.
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
    if type(a) == tuple:
        if len(a) > 1:
            a = mdot(*a)
        else:
            a = a[0]
    if type(b) == tuple:
        if len(b) > 1:
            b = mdot(*b)
        else:
            b = b[0]
    return np.dot(a, b)

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
    Li = dtrtri(L)
    Ai, _ = dpotri(L, lower=1)
    # Ai = np.tril(Ai) + np.tril(Ai,-1).T
    symmetrify(Ai)

    return Ai, L, Li, logdet


def dtrtri(L):
    """
    Inverts a Cholesky lower triangular matrix

    :param L: lower triangular matrix
    :rtype: inverse of L

    """

    L = force_F_ordered(L)
    return lapack.dtrtri(L, lower=1)[0]


def multiple_pdinv(A):
    """
    :param A: A DxDxN numpy array (each A[:,:,i] is pd)

    :rval invs: the inverses of A
    :rtype invs: np.ndarray
    :rval hld: 0.5* the log of the determinants of A
    :rtype hld: np.array

    """
    N = A.shape[-1]
    chols = [jitchol(A[:, :, i]) for i in range(N)]
    halflogdets = [np.sum(np.log(np.diag(L[0]))) for L in chols]
    invs = [dpotri(L[0], True)[0] for L in chols]
    invs = [np.triu(I) + np.triu(I, 1).T for I in invs]
    return np.dstack(invs), np.array(halflogdets)


def pca(Y, input_dim):
    """
    Principal component analysis: maximum likelihood solution by SVD

    :param Y: NxD np.array of data
    :param input_dim: int, dimension of projection


    :rval X: - Nxinput_dim np.array of dimensionality reduced data
    :rval W: - input_dimxD mapping from X to Y

    """
    if not np.allclose(Y.mean(axis=0), 0.0):
        print("Y is not zero mean, centering it locally (GPy.util.linalg.pca)")

        # Y -= Y.mean(axis=0)

    Z = linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    [X, W] = [Z[0][:, 0:input_dim], np.dot(np.diag(Z[1]), Z[2]).T[:, 0:input_dim]]
    v = X.std(axis=0)
    X /= v
    W *= v
    return X, W.T

def ppca(Y, Q, iterations=100):
    """
    EM implementation for probabilistic pca.

    :param array-like Y: Observed Data
    :param int Q: Dimensionality for reduced array
    :param int iterations: number of iterations for EM
    """
    from numpy.ma import dot as madot
    N, D = Y.shape
    # Initialise W randomly
    W = np.random.randn(D, Q) * 1e-3
    Y = np.ma.masked_invalid(Y, copy=0)
    mu = Y.mean(0)
    Ycentered = Y - mu
    try:
        for _ in range(iterations):
            exp_x = np.asarray_chkfinite(np.linalg.solve(W.T.dot(W), madot(W.T, Ycentered.T))).T
            W = np.asarray_chkfinite(np.linalg.solve(exp_x.T.dot(exp_x), madot(exp_x.T, Ycentered))).T
    except np.linalg.linalg.LinAlgError:
        #"converged"
        pass
    return np.asarray_chkfinite(exp_x), np.asarray_chkfinite(W)

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
    mat = np.asfortranarray(mat)
    out = blas.dsyrk(alpha=1.0, a=mat, beta=0.0, c=out, overwrite_c=1,
                     trans=0, lower=0)

    symmetrify(out, upper=True)
    return np.ascontiguousarray(out)

def tdot(*args, **kwargs):
    return tdot_blas(*args, **kwargs)

def DSYR_blas(A, x, alpha=1.):
    """
    Performs a symmetric rank-1 update operation:
    A <- A + alpha * np.dot(x,x.T)

    :param A: Symmetric NxN np.array
    :param x: Nx1 np.array
    :param alpha: scalar

    """
    At = blas.dsyr(lower=0, x=x, a=A, alpha=alpha, overwrite_a=False) #See https://github.com/scipy/scipy/issues/8155
    A[:] = At
    symmetrify(A, upper=True)

def DSYR_numpy(A, x, alpha=1.):
    """
    Performs a symmetric rank-1 update operation:
    A <- A + alpha * np.dot(x,x.T)

    :param A: Symmetric NxN np.array
    :param x: Nx1 np.array
    :param alpha: scalar

    """
    A += alpha * np.dot(x[:, None], x[None, :])


def DSYR(*args, **kwargs):
    return DSYR_blas(*args, **kwargs)


def symmetrify(A, upper=False):
    """
    Take the square matrix A and make it symmetrical by copting elements from
    the lower half to the upper

    works IN PLACE.

    note: tries to use cython, falls back to a slower numpy version
    """
    if config.getboolean('cython', 'working'):
        _symmetrify_cython(A, upper)
    else:
        _symmetrify_numpy(A, upper)


def _symmetrify_cython(A, upper=False):
    return linalg_cython.symmetrify(A, upper)

def _symmetrify_numpy(A, upper=False):
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]

def backsub_both_sides(L, X, transpose='left'):
    """
    Return L^-T * X * L^-1, assumuing X is symmetrical and L is lower cholesky
    """
    if transpose == 'left':
        tmp, _ = dtrtrs(L, X, lower=1, trans=1)
        return dtrtrs(L, tmp.T, lower=1, trans=1)[0].T
    else:
        tmp, _ = dtrtrs(L, X, lower=1, trans=0)
        return dtrtrs(L, tmp.T, lower=1, trans=0)[0].T

def ij_jlk_to_ilk(A, B):
    """
    Faster version of einsum 'ij,jlk->ilk'
    """
    return A.dot(B.reshape(B.shape[0], -1)).reshape(A.shape[0], B.shape[1], B.shape[2])

def ijk_jlk_to_il(A, B):
    """
    Faster version of einsum einsum('ijk,jlk->il', A,B)
    """
    res = np.zeros((A.shape[0], B.shape[1]))
    [np.add(np.dot(A[:,:,k], B[:,:,k]), res, out=res) for k in range(B.shape[-1])]
    return res

def ijk_ljk_to_ilk(A, B):
    """
    Faster version of einsum np.einsum('ijk,ljk->ilk', A, B)

    I.e A.dot(B.T) for every dimension
    """
    res = np.zeros((A.shape[-1], A.shape[0], B.shape[0]))
    [np.dot(A[:,:,i], B[:,:,i].T, out=res[i,:,:]) for i in range(A.shape[-1])]
    res = res.swapaxes(0, 2).swapaxes(0,1)
    return res
