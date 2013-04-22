# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import linalg, optimize
import pylab as pb
import Tango
import sys
import re
import pdb
import cPickle
import types
#import scipy.lib.lapack.flapack
import scipy as sp

def trace_dot(a,b):
    """
    efficiently compute the trace of the matrix product of a and b
    """
    return np.sum(a*b)

def mdot(*args):
   """Multiply all the arguments using matrix product rules.
   The output is equivalent to multiplying the arguments one by one
   from left to right using dot().
   Precedence can be controlled by creating tuples of arguments,
   for instance mdot(a,((b,c),d)) multiplies a (a*((b*c)*d)).
   Note that this means the output of dot(a,b) and mdot(a,b) will differ if
   a or b is a pure tuple of numbers.
   """
   if len(args)==1:
       return args[0]
   elif len(args)==2:
       return _mdot_r(args[0],args[1])
   else:
       return _mdot_r(args[:-1],args[-1])

def _mdot_r(a,b):
   """Recursive helper for mdot"""
   if type(a)==types.TupleType:
       if len(a)>1:
           a = mdot(*a)
       else:
           a = a[0]
   if type(b)==types.TupleType:
       if len(b)>1:
           b = mdot(*b)
       else:
           b = b[0]
   return np.dot(a,b)

def jitchol(A,maxtries=5):
    A = np.asfortranarray(A)
    L,info = linalg.lapack.flapack.dpotrf(A,lower=1)
    if info ==0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA<0.):
            raise linalg.LinAlgError, "not pd: negative diagonal elements"
        jitter= diagA.mean()*1e-6
        for i in range(1,maxtries+1):
            print '\rWarning: adding jitter of {:.10e}                        '.format(jitter),
            try:
                return linalg.cholesky(A+np.eye(A.shape[0]).T*jitter, lower = True)
            except:
                jitter *= 10
        raise linalg.LinAlgError,"not positive definite, even with jitter."



def jitchol_old(A,maxtries=5):
    """
    :param A : An almost pd square matrix

    :rval L: the Cholesky decomposition of A

    .. Note:
      Adds jitter to K, to enforce positive-definiteness
      if stuff breaks, please check:
      np.allclose(sp.linalg.cholesky(XXT, lower = True), np.triu(sp.linalg.cho_factor(XXT)[0]).T)
    """
    try:
        return linalg.cholesky(A, lower = True)
    except linalg.LinAlgError:
        diagA = np.diag(A)
        if np.any(diagA<0.):
            raise linalg.LinAlgError, "not pd: negative diagonal elements"
        jitter= diagA.mean()*1e-6
        for i in range(1,maxtries+1):
            print '\rWarning: adding jitter of {:.10e}                        '.format(jitter),
            try:
                return linalg.cholesky(A+np.eye(A.shape[0]).T*jitter, lower = True)
            except:
                jitter *= 10

        raise linalg.LinAlgError,"not positive definite, even with jitter."

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
    Ai = linalg.lapack.flapack.dpotri(L)[0]
    Ai = np.tril(Ai) + np.tril(Ai,-1).T

    return Ai, L, Li, logdet


def chol_inv(L):
    """
    Inverts a Cholesky lower triangular matrix

    :param L: lower triangular matrix
    :rtype: inverse of L

    """

    return linalg.lapack.flapack.dtrtri(L, lower = True)[0]


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
    chols = [jitchol(A[:,:,i]) for i in range(N)]
    halflogdets = [np.sum(np.log(np.diag(L[0]))) for L in chols]
    invs = [linalg.lapack.flapack.dpotri(L[0],True)[0] for L in chols]
    invs = [np.triu(I)+np.triu(I,1).T for I in invs]
    return np.dstack(invs),np.array(halflogdets)


def PCA(Y, Q):
    """
    Principal component analysis: maximum likelihood solution by SVD

    Arguments
    ---------
    :param Y: NxD np.array of data
    :param Q: int, dimension of projection

    Returns
    -------
    :rval X: - NxQ np.array of dimensionality reduced data
    W - QxD mapping from X to Y
    """
    if not np.allclose(Y.mean(axis=0), 0.0):
        print "Y is not zero mean, centering it locally (GPy.util.linalg.PCA)"
        
        #Y -= Y.mean(axis=0) 

    Z = linalg.svd(Y-Y.mean(axis=0), full_matrices = False)
    [X, W] = [Z[0][:,0:Q], np.dot(np.diag(Z[1]), Z[2]).T[:,0:Q]]
    v = X.std(axis=0)
    X /= v;
    W *= v;
    return X, W.T
