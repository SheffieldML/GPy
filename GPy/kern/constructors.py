# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from kern import kern
import parts


def rbf_inv(input_dim,variance=1., inv_lengthscale=None,ARD=False):
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
    part = parts.rbf_inv.RBFInv(input_dim,variance,inv_lengthscale,ARD)
    return kern(input_dim, [part])

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
    part = parts.rbf.RBF(input_dim,variance,lengthscale,ARD)
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
    part = parts.linear.Linear(input_dim,variances,ARD)
    return kern(input_dim, [part])

def mlp(input_dim,variance=1., weight_variance=None,bias_variance=100.,ARD=False):
    """
    Construct an MLP kernel

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param weight_scale: the lengthscale of the kernel
    :type weight_scale: vector of weight variances for input weights in neural network (length 1 if kernel is isotropic)
    :param bias_variance: the variance of the biases in the neural network.
    :type bias_variance: float
    :param ARD: Auto Relevance Determination (allows for ARD version of covariance)
    :type ARD: Boolean
    """
    part = parts.mlp.MLP(input_dim,variance,weight_variance,bias_variance,ARD)
    return kern(input_dim, [part])

def white(input_dim,variance=1.):
    """
     Construct a white kernel.

     Arguments
     ---------
    input_dimD (int), obligatory
     variance (float)
    """
    part = parts.white.White(input_dim,variance)
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
    part = parts.exponential.Exponential(input_dim,variance, lengthscale, ARD)
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
    part = parts.Matern32.Matern32(input_dim,variance, lengthscale, ARD)
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
    part = parts.Matern52.Matern52(input_dim, variance, lengthscale, ARD)
    return kern(input_dim, [part])

def bias(input_dim, variance=1.):
    """
     Construct a bias kernel.

     Arguments
     ---------
     input_dim (int), obligatory
     variance (float)
    """
    part = parts.bias.Bias(input_dim, variance)
    return kern(input_dim, [part])

def finite_dimensional(input_dim, F, G, variances=1., weights=None):
    """
    Construct a finite dimensional kernel.
    input_dim: int - the number of input dimensions
    F: np.array of functions with shape (n,) - the n basis functions
    G: np.array with shape (n,n) - the Gram matrix associated to F
    variances : np.ndarray with shape (n,)
    """
    part = parts.finite_dimensional.FiniteDimensional(input_dim, F, G, variances, weights)
    return kern(input_dim, [part])

def spline(input_dim, variance=1.):
    """
    Construct a spline kernel.

    :param input_dim: Dimensionality of the kernel
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    """
    part = parts.spline.Spline(input_dim, variance)
    return kern(input_dim, [part])

def Brownian(input_dim, variance=1.):
    """
    Construct a Brownian motion kernel.

    :param input_dim: Dimensionality of the kernel
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    """
    part = parts.Brownian.Brownian(input_dim, variance)
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
    part = parts.periodic_exponential.PeriodicExponential(input_dim, variance, lengthscale, period, n_freq, lower, upper)
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
    part = parts.periodic_Matern32.PeriodicMatern32(input_dim, variance, lengthscale, period, n_freq, lower, upper)
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
    part = parts.periodic_Matern52.PeriodicMatern52(input_dim, variance, lengthscale, period, n_freq, lower, upper)
    return kern(input_dim, [part])

def prod(k1,k2,tensor=False):
    """
     Construct a product kernel over input_dim from two kernels over input_dim

    :param k1, k2: the kernels to multiply
    :type k1, k2: kernpart
    :param tensor: The kernels are either multiply as functions defined on the same input space (default) or on the product of the input spaces
    :type tensor: Boolean
    :rtype: kernel object
    """
    part = parts.prod.Prod(k1, k2, tensor)
    return kern(part.input_dim, [part])

def symmetric(k):
    """
    Construct a symmetric kernel from an existing kernel
    """
    k_ = k.copy()
    k_.parts = [symmetric.Symmetric(p) for p in k.parts]
    return k_

def coregionalise(Nout, R=1, W=None, kappa=None):
    p = parts.coregionalise.Coregionalise(Nout,R,W,kappa)
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
    part = parts.rational_quadratic.RationalQuadratic(input_dim, variance, lengthscale, power)
    return kern(input_dim, [part])

def fixed(input_dim, K, variance=1.):
    """
     Construct a Fixed effect kernel.

    :param input_dim: the number of input dimensions
    :type input_dim: int (input_dim=1 is the only value currently supported)
    :param K: the variance :math:`\sigma^2`
    :type K: np.array
    :param variance: kernel variance
    :type variance: float
    :rtype: kern object
    """
    part = parts.fixed.Fixed(input_dim, K, variance)
    return kern(input_dim, [part])

def rbfcos(input_dim, variance=1., frequencies=None, bandwidths=None, ARD=False):
    """
    construct a rbfcos kernel
    """
    part = parts.rbfcos.RBFCos(input_dim, variance, frequencies, bandwidths, ARD)
    return kern(input_dim, [part])

def independent_outputs(k):
    """
    Construct a kernel with independent outputs from an existing kernel
    """
    for sl in k.input_slices:
        assert (sl.start is None) and (sl.stop is None), "cannot adjust input slices! (TODO)"
    _parts = [parts.independent_outputs.IndependentOutputs(p) for p in k.parts]
    return kern(k.input_dim+1,_parts)

def hierarchical(k):
    """
    TODO THis can't be right! Construct a kernel with independent outputs from an existing kernel
    """
    # for sl in k.input_slices:
    #     assert (sl.start is None) and (sl.stop is None), "cannot adjust input slices! (TODO)"
    _parts = [parts.hierarchical.Hierarchical(k.parts)]
    return kern(k.input_dim+len(k.parts),_parts)
