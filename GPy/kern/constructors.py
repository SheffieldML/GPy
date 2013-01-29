# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from kern import kern

from rbf import rbf as rbfpart
from white import white as whitepart
from linear import linear as linearpart
from exponential import exponential as exponentialpart
from Matern32 import Matern32 as Matern32part
from Matern52 import Matern52 as Matern52part
from bias import bias as biaspart
from finite_dimensional import finite_dimensional as finite_dimensionalpart
from spline import spline as splinepart
from Brownian import Brownian as Brownianpart
from periodic_exponential import periodic_exponential as periodic_exponentialpart
from periodic_Matern32 import periodic_Matern32 as periodic_Matern32part
from periodic_Matern52 import periodic_Matern52 as periodic_Matern52part

#TODO these s=constructors are not as clean as we'd like. Tidy the code up
#using meta-classes to make the objects construct properly wthout them.


def rbf(D,variance=1., lengthscale=None,ARD=False):
    """
    Construct an RBF kernel

    :param D: dimensionality of the kernel, obligatory
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = rbfpart(D,variance,lengthscale,ARD)
    return kern(D, [part])

def linear(D,variances=None,ARD=True):
    """
     Construct a linear kernel.

     Arguments
     ---------
     D (int), obligatory
     variances (np.ndarray)
     ARD (boolean)
    """
    part = linearpart(D,variances,ARD)
    return kern(D, [part])

def white(D,variance=1.):
    """
     Construct a white kernel.

     Arguments
     ---------
     D (int), obligatory
     variance (float)
    """
    part = whitepart(D,variance)
    return kern(D, [part])

def exponential(D,variance=1., lengthscale=None, ARD=False):
    """
    Construct an exponential kernel

    :param D: dimensionality of the kernel, obligatory
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = exponentialpart(D,variance, lengthscale, ARD)
    return kern(D, [part])

def Matern32(D,variance=1., lengthscale=None, ARD=False):
    """
     Construct a Matern 3/2 kernel.

    :param D: dimensionality of the kernel, obligatory
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = Matern32part(D,variance, lengthscale, ARD)
    return kern(D, [part])

def Matern52(D,variance=1., lengthscale=None, ARD=False):
    """
     Construct a Matern 5/2 kernel.

    :param D: dimensionality of the kernel, obligatory
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = Matern52part(D,variance, lengthscale, ARD)
    return kern(D, [part])

def bias(D,variance=1.):
    """
     Construct a bias kernel.

     Arguments
     ---------
     D (int), obligatory
     variance (float)
    """
    part = biaspart(D,variance)
    return kern(D, [part])

def finite_dimensional(D,F,G,variances=1.,weights=None):
    """
    Construct a finite dimensional kernel.
    D: int - the number of input dimensions
    F: np.array of functions with shape (n,) - the n basis functions
    G: np.array with shape (n,n) - the Gram matrix associated to F
    variances : np.ndarray with shape (n,)
    """
    part = finite_dimensionalpart(D,F,G,variances,weights)
    return kern(D, [part])

def spline(D,variance=1.):
    """
    Construct a spline kernel.

    :param D: Dimensionality of the kernel
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    """
    part = splinepart(D,variance)
    return kern(D, [part])

def Brownian(D,variance=1.):
    """
    Construct a Brownian motion kernel.

    :param D: Dimensionality of the kernel
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    """
    part = Brownianpart(D,variance)
    return kern(D, [part])

import sympy as sp
from sympykern import spkern
from sympy.parsing.sympy_parser import parse_expr

def rbf_sympy(D,ARD=False,variance=1., lengthscale=1.):
    """
    Radial Basis Function covariance.
    """
    X = [sp.var('x%i'%i) for i in range(D)]
    Z = [sp.var('z%i'%i) for i in range(D)]
    rbf_variance = sp.var('rbf_variance',positive=True)
    if ARD:
        rbf_lengthscales = [sp.var('rbf_lengthscale_%i'%i,positive=True) for i in range(D)]
        dist_string = ' + '.join(['(x%i-z%i)**2/rbf_lengthscale_%i**2'%(i,i,i) for i in range(D)])
        dist = parse_expr(dist_string)
        f =  rbf_variance*sp.exp(-dist/2.)
    else:
        rbf_lengthscale = sp.var('rbf_lengthscale',positive=True)
        dist_string = ' + '.join(['(x%i-z%i)**2'%(i,i) for i in range(D)])
        dist = parse_expr(dist_string)
        f =  rbf_variance*sp.exp(-dist/(2*rbf_lengthscale**2))
    return kern(D,[spkern(D,f)])

def sympykern(D,k):
    """
    A kernel from a symbolic sympy representation
    """
    return kern(D,[spkern(D,k)])

def periodic_exponential(D=1,variance=1., lengthscale=None, period=2*np.pi,n_freq=10,lower=0.,upper=4*np.pi):
    """
    Construct an periodic exponential kernel

    :param D: dimensionality, only defined for D=1
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param period: the period
    :type period: float
    :param n_freq: the number of frequencies considered for the periodic subspace
    :type n_freq: int
    """
    part = periodic_exponentialpart(D,variance, lengthscale, period, n_freq, lower, upper)
    return kern(D, [part])

def periodic_Matern32(D,variance=1., lengthscale=None, period=2*np.pi,n_freq=10,lower=0.,upper=4*np.pi):
    """
     Construct a periodic Matern 3/2 kernel.

     :param D: dimensionality, only defined for D=1
     :type D: int
     :param variance: the variance of the kernel
     :type variance: float
     :param lengthscale: the lengthscale of the kernel
     :type lengthscale: float
     :param period: the period
     :type period: float
     :param n_freq: the number of frequencies considered for the periodic subspace
     :type n_freq: int
    """
    part = periodic_Matern32part(D,variance, lengthscale, period, n_freq, lower, upper)
    return kern(D, [part])

def periodic_Matern52(D,variance=1., lengthscale=None, period=2*np.pi,n_freq=10,lower=0.,upper=4*np.pi):
    """
     Construct a periodic Matern 5/2 kernel.

     :param D: dimensionality, only defined for D=1
     :type D: int
     :param variance: the variance of the kernel
     :type variance: float
     :param lengthscale: the lengthscale of the kernel
     :type lengthscale: float
     :param period: the period
     :type period: float
     :param n_freq: the number of frequencies considered for the periodic subspace
     :type n_freq: int
    """
    part = periodic_Matern52part(D,variance, lengthscale, period, n_freq, lower, upper)
    return kern(D, [part])
