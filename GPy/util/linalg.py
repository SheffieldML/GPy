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
import warnings

if np.all(np.float64((scipy.__version__).split('.')[:2]) >= np.array([0, 12])):
    import scipy.linalg.lapack as lapack
else:
    from scipy.linalg.lapack import flapack as lapack

try:
    _blaslib = ctypes.cdll.LoadLibrary(np.core._dotblas.__file__) # @UndefinedVariable
    _blas_available = True
    assert hasattr(_blaslib, 'dsyrk_')
    assert hasattr(_blaslib, 'dsyr_')
except AssertionError:
    _blas_available = False
except AttributeError as e:
    _blas_available = False
    warnings.warn("warning: caught this exception:" + str(e))

def dtrtrs(A, B, lower=0, trans=0, unitdiag=0):
    """
    Wrapper for lapack dtrtrs function

    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:

    """
    return lapack.dtrtrs(A, B, lower=lower, trans=trans, unitdiag=unitdiag)

def dpotrs(A, B, lower=0):
    """
    Wrapper for lapack dpotrs function

    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:

    """
    return lapack.dpotrs(A, B, lower=lower)

def dpotri(A, lower=0):
    """
    Wrapper for lapack dpotri function

    :param A: Matrix A
    :param lower: is matrix lower (true) or upper (false)
    :returns: A inverse

    """
    return lapack.dpotri(A, lower=lower)

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
    return np.sum(a * b)

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
    :param A: An almost pd square matrix

    :rval L: the Cholesky decomposition of A

    .. note:

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
    :param A: A DxDxN numpy array (each A[:,:,i] is pd)

    :rval invs: the inverses of A
    :rtype invs: np.ndarray
    :rval hld: 0.5* the log of the determinants of A
    :rtype hld: np.array

    """
    N = A.shape[-1]
    chols = [jitchol(A[:, :, i]) for i in range(N)]
    halflogdets = [np.sum(np.log(np.diag(L[0]))) for L in chols]
    invs = [lapack.dpotri(L[0], True)[0] for L in chols]
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
        print "Y is not zero mean, centering it locally (GPy.util.linalg.pca)"

        # Y -= Y.mean(axis=0)

    Z = linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    [X, W] = [Z[0][:, 0:input_dim], np.dot(np.diag(Z[1]), Z[2]).T[:, 0:input_dim]]
    v = X.std(axis=0)
    X /= v;
    W *= v;
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

def ppca_missing_data_at_random(Y, Q, iters=100):
    """
    EM implementation of Probabilistic pca for when there is missing data.
    
    Taken from <SheffieldML, https://github.com/SheffieldML>

    .. math:
        \\mathbf{Y} = \mathbf{XW} + \\epsilon \\text{, where}
        \\epsilon = \\mathcal{N}(0, \\sigma^2 \mathbf{I})
        
    :returns: X, W, sigma^2 
    """
    from numpy.ma import dot as madot
    import diag
    from GPy.util.subarray_and_sorting import common_subarrays
    import time
    debug = 1
    # Initialise W randomly
    N, D = Y.shape
    W = np.random.randn(Q, D) * 1e-3
    Y = np.ma.masked_invalid(Y, copy=1)
    nu = 1.
    #num_obs_i = 1./Y.count()
    Ycentered = Y - Y.mean(0)
    
    X = np.zeros((N,Q))
    cs = common_subarrays(Y.mask)
    cr = common_subarrays(Y.mask, 1)
    Sigma = np.zeros((N, Q, Q))
    Sigma2 = np.zeros((N, Q, Q))
    mu = np.zeros(D)
    if debug:
        import matplotlib.pyplot as pylab
        fig = pylab.figure("FIT MISSING DATA"); 
        ax = fig.gca()
        ax.cla()
        lines = pylab.plot(np.zeros((N,Q)).dot(W))
    W2 = np.zeros((Q,D))

    for i in range(iters):
#         Sigma = np.linalg.solve(diag.add(madot(W,W.T), nu), diag.times(np.eye(Q),nu))
#         exp_x = madot(madot(Ycentered, W.T),Sigma)/nu
#         Ycentered = (Y - exp_x.dot(W).mean(0))
#         #import ipdb;ipdb.set_trace()
#         #Ycentered = mu
#         W = np.linalg.solve(madot(exp_x.T,exp_x) + Sigma, madot(exp_x.T, Ycentered))
#         nu = (((Ycentered - madot(exp_x, W))**2).sum(0) + madot(W.T,madot(Sigma,W)).sum(0)).sum()/N
        for csi, (mask, index) in enumerate(cs.iteritems()):
            mask = ~np.array(mask)
            Sigma2[index, :, :] = nu * np.linalg.inv(diag.add(W2[:,mask].dot(W2[:,mask].T), nu))
            #X[index,:] = madot((Sigma[csi]/nu),madot(W,Ycentered[index].T))[:,0]
        X2 = ((Sigma2/nu) * (madot(Ycentered,W2.T).base)[:,:,None]).sum(-1)
        mu2 = (Y - X.dot(W)).mean(0)
        for n in range(N):
            Sigma[n] = nu * np.linalg.inv(diag.add(W[:,~Y.mask[n]].dot(W[:,~Y.mask[n]].T), nu))
            X[n, :] = (Sigma[n]/nu).dot(W[:,~Y.mask[n]].dot(Ycentered[n,~Y.mask[n]].T))
        for d in range(D):
            mu[d] = (Y[~Y.mask[:,d], d] - X[~Y.mask[:,d]].dot(W[:, d])).mean()
        Ycentered = (Y - mu)
        nu3 = 0.
        for cri, (mask, index) in enumerate(cr.iteritems()):
            mask = ~np.array(mask)
            W2[:,index] = np.linalg.solve(X[mask].T.dot(X[mask]) + Sigma[mask].sum(0), madot(X[mask].T, Ycentered[mask,index]))[:,None]
            W2[:,index] = np.linalg.solve(X.T.dot(X) + Sigma.sum(0), madot(X.T, Ycentered[:,index]))
            #nu += (((Ycentered[mask,index] - X[mask].dot(W[:,index]))**2).sum(0) + W[:,index].T.dot(Sigma[mask].sum(0).dot(W[:,index])).sum(0)).sum()
            nu3 += (((Ycentered[index] - X.dot(W[:,index]))**2).sum(0) + W[:,index].T.dot(Sigma.sum(0).dot(W[:,index])).sum(0)).sum()
        nu3 /= N
        nu = 0.
        nu2 = 0.
        W = np.zeros((Q,D))
        for j in range(D):
            W[:,j] = np.linalg.solve(X[~Y.mask[:,j]].T.dot(X[~Y.mask[:,j]]) + Sigma[~Y.mask[:,j]].sum(0), madot(X[~Y.mask[:,j]].T, Ycentered[~Y.mask[:,j],j]))
            nu2f = np.tensordot(W[:,j].T, Sigma[~Y.mask[:,j],:,:], [0,1]).dot(W[:,j])
            nu2s = W[:,j].T.dot(Sigma[~Y.mask[:,j],:,:].sum(0).dot(W[:,j]))
            nu2 += (((Ycentered[~Y.mask[:,j],j] - X[~Y.mask[:,j],:].dot(W[:,j]))**2) + nu2f).sum()
            for i in range(N):
                if not Y.mask[i,j]:
                    nu += ((Ycentered[i,j] - X[i,:].dot(W[:,j]))**2) + W[:,j].T.dot(Sigma[i,:,:].dot(W[:,j]))
        nu /= N
        nu2 /= N
        nu4 = (((Ycentered - X.dot(W))**2).sum(0) + W.T.dot(Sigma.sum(0).dot(W)).sum(0)).sum()/N
        import ipdb;ipdb.set_trace()
        if debug:
            #print Sigma[0]
            print "nu:", nu, "sum(X):", X.sum()
            pred_y = X.dot(W)
            for x, l in zip(pred_y.T, lines):
                l.set_ydata(x)
            ax.autoscale_view()
            ax.set_ylim(pred_y.min(), pred_y.max())
            fig.canvas.draw()
            time.sleep(.3)
    return np.asarray_chkfinite(X), np.asarray_chkfinite(W), nu


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

    N = int(N) # for safe type casting
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

    if L is the lower chol. of K, then this function computes L\_
    where L\_ is the lower chol of K + x*x^T

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
