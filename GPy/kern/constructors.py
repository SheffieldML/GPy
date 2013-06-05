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
from fixed import Fixed as fixedpart
from finite_dimensional import finite_dimensional as finite_dimensionalpart
from spline import spline as splinepart
from Brownian import Brownian as Brownianpart
from periodic_exponential import periodic_exponential as periodic_exponentialpart
from periodic_Matern32 import periodic_Matern32 as periodic_Matern32part
from periodic_Matern52 import periodic_Matern52 as periodic_Matern52part
from prod import prod as prodpart
from symmetric import symmetric as symmetric_part
from coregionalise import Coregionalise as coregionalise_part
from rational_quadratic import rational_quadratic as rational_quadraticpart
from rbfcos import rbfcos as rbfcospart
from independent_outputs import IndependentOutputs as independent_output_part
#TODO these s=constructors are not as clean as we'd like. Tidy the code up
#using meta-classes to make the objects construct properly wthout them.


def rbf(input_dim,variance=1., lengthscale=None,ARD=False):
    """
    Construct an RBF kernel

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = rbfpart(input_dim,variance,lengthscale,ARD)
    return kern(input_dim, [part])

def linear(input_dim,variances=None,ARD=False):
    """
     Construct a linear kernel.

     Arguments
     ---------
    input_dimD (int), obligatory
     variances (np.ndarray)
     ARD (boolean)
    """
    part = linearpart(input_dim,variances,ARD)
    return kern(input_dim, [part])

def white(input_dim,variance=1.):
    """
     Construct a white kernel.

     Arguments
     ---------
    input_dimD (int), obligatory
     variance (float)
    """
    part = whitepart(input_dim,variance)
    return kern(input_dim, [part])

def exponential(input_dim,variance=1., lengthscale=None, ARD=False):
    """
    Construct an exponential kernel

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = exponentialpart(input_dim,variance, lengthscale, ARD)
    return kern(input_dim, [part])

def Matern32(input_dim,variance=1., lengthscale=None, ARD=False):
    """
     Construct a Matern 3/2 kernel.

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = Matern32part(input_dim,variance, lengthscale, ARD)
    return kern(input_dim, [part])

def Matern52(input_dim, variance=1., lengthscale=None, ARD=False):
    """
     Construct a Matern 5/2 kernel.

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean
    """
    part = Matern52part(input_dim, variance, lengthscale, ARD)
    return kern(input_dim, [part])

def bias(input_dim, variance=1.):
    """
     Construct a bias kernel.

     Arguments
     ---------
     input_dim (int), obligatory
     variance (float)
    """
    part = biaspart(input_dim, variance)
    return kern(input_dim, [part])

def finite_dimensional(input_dim, F, G, variances=1., weights=None):
    """
    Construct a finite dimensional kernel.
    input_dim: int - the number of input dimensions
    F: np.array of functions with shape (n,) - the n basis functions
    G: np.array with shape (n,n) - the Gram matrix associated to F
    variances : np.ndarray with shape (n,)
    """
    part = finite_dimensionalpart(input_dim, F, G, variances, weights)
    return kern(input_dim, [part])

def spline(input_dim, variance=1.):
    """
    Construct a spline kernel.

    :param input_dim: Dimensionality of the kernel
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    """
    part = splinepart(input_dim, variance)
    return kern(input_dim, [part])

def Brownian(input_dim, variance=1.):
    """
    Construct a Brownian motion kernel.

    :param input_dim: Dimensionality of the kernel
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    """
    part = Brownianpart(input_dim, variance)
    return kern(input_dim, [part])

try:
    import sympy as sp
    from sympykern import spkern
    from sympy.parsing.sympy_parser import parse_expr
    sympy_available = True
except ImportError:
    sympy_available = False

if sympy_available:
    def rbf_sympy(input_dim, ARD=False, variance=1., lengthscale=1.):
        """
        Radial Basis Function covariance.
        """
        X = [sp.var('x%i' % i) for i in range(input_dim)]
        Z = [sp.var('z%i' % i) for i in range(input_dim)]
        rbf_variance = sp.var('rbf_variance',positive=True)
        if ARD:
            rbf_lengthscales = [sp.var('rbf_lengthscale_%i' % i, positive=True) for i in range(input_dim)]
            dist_string = ' + '.join(['(x%i-z%i)**2/rbf_lengthscale_%i**2' % (i, i, i) for i in range(input_dim)])
            dist = parse_expr(dist_string)
            f =  rbf_variance*sp.exp(-dist/2.)
        else:
            rbf_lengthscale = sp.var('rbf_lengthscale',positive=True)
            dist_string = ' + '.join(['(x%i-z%i)**2' % (i, i) for i in range(input_dim)])
            dist = parse_expr(dist_string)
            f =  rbf_variance*sp.exp(-dist/(2*rbf_lengthscale**2))
        return kern(input_dim, [spkern(input_dim, f)])

    def sympykern(input_dim, k):
        """
        A kernel from a symbolic sympy representation
        """
        return kern(input_dim, [spkern(input_dim, k)])
del sympy_available

def periodic_exponential(input_dim=1, variance=1., lengthscale=None, period=2 * np.pi, n_freq=10, lower=0., upper=4 * np.pi):
    """
    Construct an periodic exponential kernel

    :param input_dim: dimensionality, only defined for input_dim=1
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the lengthscale of the kernel
    :type lengthscale: float
    :param period: the period
    :type period: float
    :param n_freq: the number of frequencies considered for the periodic subspace
    :type n_freq: int
    """
    part = periodic_exponentialpart(input_dim, variance, lengthscale, period, n_freq, lower, upper)
    return kern(input_dim, [part])

def periodic_Matern32(input_dim, variance=1., lengthscale=None, period=2 * np.pi, n_freq=10, lower=0., upper=4 * np.pi):
    """
     Construct a periodic Matern 3/2 kernel.

     :param input_dim: dimensionality, only defined for input_dim=1
     :type input_dim: int
     :param variance: the variance of the kernel
     :type variance: float
     :param lengthscale: the lengthscale of the kernel
     :type lengthscale: float
     :param period: the period
     :type period: float
     :param n_freq: the number of frequencies considered for the periodic subspace
     :type n_freq: int
    """
    part = periodic_Matern32part(input_dim, variance, lengthscale, period, n_freq, lower, upper)
    return kern(input_dim, [part])

def periodic_Matern52(input_dim, variance=1., lengthscale=None, period=2 * np.pi, n_freq=10, lower=0., upper=4 * np.pi):
    """
     Construct a periodic Matern 5/2 kernel.

     :param input_dim: dimensionality, only defined for input_dim=1
     :type input_dim: int
     :param variance: the variance of the kernel
     :type variance: float
     :param lengthscale: the lengthscale of the kernel
     :type lengthscale: float
     :param period: the period
     :type period: float
     :param n_freq: the number of frequencies considered for the periodic subspace
     :type n_freq: int
    """
    part = periodic_Matern52part(input_dim, variance, lengthscale, period, n_freq, lower, upper)
    return kern(input_dim, [part])

def prod(k1,k2,tensor=False):
    """
     Construct a product kernel over input_dim from two kernels over input_dim

    :param k1, k2: the kernels to multiply
    :type k1, k2: kernpart
    :rtype: kernel object
    """
    part = prodpart(k1,k2,tensor)
    return kern(part.input_dim, [part])

def symmetric(k):
    """
    Construct a symmetrical kernel from an existing kernel
    """
    k_ = k.copy()
    k_.parts = [symmetric_part(p) for p in k.parts]
    return k_

def Coregionalise(Nout,R=1, W=None, kappa=None):
    p = coregionalise_part(Nout,R,W,kappa)
    return kern(1,[p])


def rational_quadratic(input_dim, variance=1., lengthscale=1., power=1.):
    """
     Construct rational quadratic kernel.

    :param input_dim: the number of input dimensions
    :type input_dim: int (input_dim=1 is the only value currently supported)
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param lengthscale: the lengthscale :math:`\ell`
    :type lengthscale: float
    :rtype: kern object

    """
    part = rational_quadraticpart(input_dim, variance, lengthscale, power)
    return kern(input_dim, [part])

def Fixed(input_dim, K, variance=1.):
    """
     Construct a Fixed effect kernel.

     Arguments
     ---------
     input_dim (int), obligatory
     K (np.array), obligatory
     variance (float)
    """
    part = fixedpart(input_dim, K, variance)
    return kern(input_dim, [part])

def rbfcos(input_dim, variance=1., frequencies=None, bandwidths=None, ARD=False):
    """
    construct a rbfcos kernel
    """
    part = rbfcospart(input_dim, variance, frequencies, bandwidths, ARD)
    return kern(input_dim, [part])

def IndependentOutputs(k):
    """
    Construct a kernel with independent outputs from an existing kernel
    """
    for sl in k.input_slices:
        assert (sl.start is None) and (sl.stop is None), "cannot adjust input slices! (TODO)"
    parts = [independent_output_part(p) for p in k.parts]
    return kern(k.input_dim+1,parts)


