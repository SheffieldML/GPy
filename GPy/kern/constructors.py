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
from fixed import fixed as fixedpart
from finite_dimensional import finite_dimensional as finite_dimensionalpart
from spline import spline as splinepart
from Brownian import Brownian as Brownianpart
from periodic_exponential import periodic_exponential as periodic_exponentialpart
from periodic_Matern32 import periodic_Matern32 as periodic_Matern32part
from periodic_Matern52 import periodic_Matern52 as periodic_Matern52part
from prod import prod as prodpart
from prod_orthogonal import prod_orthogonal as prod_orthogonalpart
from symmetric import symmetric as symmetric_part
from coregionalise import coregionalise as coregionalise_part
from rational_quadratic import rational_quadratic as rational_quadraticpart
from rbfcos import rbfcos as rbfcospart
from independent_outputs import independent_outputs as independent_output_part
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

def linear(D,variances=None,ARD=False):
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

try:
    import sympy as sp
    from sympykern import spkern
    from sympy.parsing.sympy_parser import parse_expr
    sympy_available = True
except ImportError:
    sympy_available = False

if sympy_available:
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
del sympy_available

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

def prod(k1,k2):
    """
     Construct a product kernel over D from two kernels over D

    :param k1, k2: the kernels to multiply
    :type k1, k2: kernpart
    :rtype: kernel object
    """
    part = prodpart(k1,k2)
    return kern(k1.D, [part])

def prod_orthogonal(k1,k2):
    """
     Construct a product kernel over D1 x D2 from a kernel over D1 and another over D2.

    :param k1, k2: the kernels to multiply
    :type k1, k2: kernpart
    :rtype: kernel object
    """
    part = prod_orthogonalpart(k1,k2)
    return kern(k1.D+k2.D, [part])

def symmetric(k):
    """
    Construct a symmetrical kernel from an existing kernel
    """
    k_ = k.copy()
    k_.parts = [symmetric_part(p) for p in k.parts]
    return k_

def coregionalise(Nout,R=1, W=None, kappa=None):
    p = coregionalise_part(Nout,R,W,kappa)
    return kern(1,[p])


def rational_quadratic(D,variance=1., lengthscale=1., power=1.):
    """
     Construct rational quadratic kernel.

    :param D: the number of input dimensions
    :type D: int (D=1 is the only value currently supported)
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param lengthscale: the lengthscale :math:`\ell`
    :type lengthscale: float
    :rtype: kern object

    """
    part = rational_quadraticpart(D,variance, lengthscale, power)
    return kern(D, [part])

def fixed(D, K, variance=1.):
    """
     Construct a fixed effect kernel.

     Arguments
     ---------
     D (int), obligatory
     K (np.array), obligatory
     variance (float)
    """
    part = fixedpart(D, K, variance)
    return kern(D, [part])

def rbfcos(D,variance=1.,frequencies=None,bandwidths=None,ARD=False):
    """
    construct a rbfcos kernel
    """
    part = rbfcospart(D,variance,frequencies,bandwidths,ARD)
    return kern(D,[part])

def independent_outputs(k):
    """
    Construct a kernel with independent outputs from an existing kernel
    """
    for sl in k.input_slices:
        assert (sl.start is None) and (sl.stop is None), "cannot adjust input slices! (TODO)"
    parts = [independent_output_part(p) for p in k.parts]
    return kern(k.D+1,parts)


