# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from kern import kern

from rbf import rbf as rbfpart
from rbf_ARD import rbf_ARD as rbf_ARD_part
from white import white as whitepart
from linear import linear as linearpart
from linear_ARD import linear_ARD as linear_ARD_part
from exponential import exponential as exponentialpart
from Matern32 import Matern32 as Matern32part
from Matern52 import Matern52 as Matern52part
from bias import bias as biaspart
from finite_dimensional import finite_dimensional as finite_dimensionalpart
from spline import spline as splinepart
from Brownian import Brownian as Brownianpart

#TODO these s=constructors are not as clean as we'd like. Tidy the code up
#using meta-classes to make the objects construct properly wthout them.


def rbf(D,variance=1., lengthscale=1.):
    """
    Construct an RBF kernel

    :param D: dimensionality of the kernel, obligatory
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    """
    part = rbfpart(D,variance,lengthscale)
    return kern(D, [part])

def rbf_ARD(D,variance=1., lengthscales=None):
    """
    Construct an RBF kernel with Automatic Relevance Determination (ARD)

    :param D: dimensionality of the kernel, obligatory
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscales: the lengthscales of the kernel
    :type lengthscales: None|np.ndarray
    """
    part = rbf_ARD_part(D,variance,lengthscales)
    return kern(D, [part])

def linear(D,lengthscales=None):
    """
     Construct a linear kernel.

     Arguments
     ---------
     D (int), obligatory
     lengthscales (np.ndarray)
    """
    part = linearpart(D,lengthscales)
    return kern(D, [part])

def linear_ARD(D,lengthscales=None):
    """
     Construct a linear ARD kernel.

     Arguments
     ---------
     D (int), obligatory
     lengthscales (np.ndarray)
    """
    part = linear_ARD_part(D,lengthscales)
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

def exponential(D,variance=1., lengthscales=None):
    """
     Construct a exponential kernel.

     Arguments
     ---------
     D (int), obligatory
     variance (float)
     lengthscales (np.ndarray)
    """
    part = exponentialpart(D,variance, lengthscales)
    return kern(D, [part])

def Matern32(D,variance=1., lengthscales=None):
    """
     Construct a Matern 3/2 kernel.

     Arguments
     ---------
     D (int), obligatory
     variance (float)
     lengthscales (np.ndarray)
    """
    part = Matern32part(D,variance, lengthscales)
    return kern(D, [part])

def Matern52(D,variance=1., lengthscales=None):
    """
     Construct a Matern 5/2 kernel.

     Arguments
     ---------
     D (int), obligatory
     variance (float)
     lengthscales (np.ndarray)
    """
    part = Matern52part(D,variance, lengthscales)
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

def rbf_sympy(D,variance=1., lengthscale=1.):
    """
    Radial Basis Function covariance.
    """
    X = [sp.var('x%i'%i) for i in range(D)]
    Z = [sp.var('z%i'%i) for i in range(D)]
    rbf_variance = sp.var('rbf_variance',positive=True)
    rbf_lengthscale = sp.var('rbf_lengthscale',positive=True)
    dist_string = ' + '.join(['(x%i-z%i)**2'%(i,i) for i in range(D)])
    dist = parse_expr(dist_string)
    f =  rbf_variance*sp.exp(-dist/(2*rbf_lengthscale**2))
    return kern(D,[spkern(D,f,np.array([variance,lengthscale]))])

def sympykern(D,k):
    """
    A kernel from a symbolic sympy representation
    """
    return kern(D,[spkern(D,k)])
