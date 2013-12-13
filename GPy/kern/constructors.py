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

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variances:
    :type variances: np.ndarray
    :param ARD: Auto Relevance Determination (one lengthscale per dimension)
    :type ARD: Boolean

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

def gibbs(input_dim,variance=1., mapping=None):
    """

    Gibbs and MacKay non-stationary covariance function.

    .. math::

       r = \\sqrt{((x_i - x_j)'*(x_i - x_j))}

       k(x_i, x_j) = \\sigma^2*Z*exp(-r^2/(l(x)*l(x) + l(x')*l(x')))

       Z = \\sqrt{2*l(x)*l(x')/(l(x)*l(x) + l(x')*l(x')}

    Where :math:`l(x)` is a function giving the length scale as a function of space.

    This is the non stationary kernel proposed by Mark Gibbs in his 1997
    thesis. It is similar to an RBF but has a length scale that varies
    with input location. This leads to an additional term in front of
    the kernel.

    The parameters are :math:`\\sigma^2`, the process variance, and the parameters of l(x) which is a function that can be specified by the user, by default an multi-layer peceptron is used is used.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\\sigma^2`
    :type variance: float
    :param mapping: the mapping that gives the lengthscale across the input space.
    :type mapping: GPy.core.Mapping
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one weight variance parameter :math:`\\sigma^2_w`), otherwise there is one weight variance parameter per dimension.
    :type ARD: Boolean
    :rtype: Kernpart object

    """
    part = parts.gibbs.Gibbs(input_dim,variance,mapping)
    return kern(input_dim, [part])

def hetero(input_dim, mapping=None, transform=None):
    """
    """
    part = parts.hetero.Hetero(input_dim,mapping,transform)
    return kern(input_dim, [part])

def poly(input_dim,variance=1., weight_variance=None,bias_variance=1.,degree=2, ARD=False):
    """
    Construct a polynomial kernel

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param weight_scale: the lengthscale of the kernel
    :type weight_scale: vector of weight variances for input weights.
    :param bias_variance: the variance of the biases.
    :type bias_variance: float
    :param degree: the degree of the polynomial
    :type degree: int
    :param ARD: Auto Relevance Determination (allows for ARD version of covariance)
    :type ARD: Boolean

    """
    part = parts.poly.POLY(input_dim,variance,weight_variance,bias_variance,degree,ARD)
    return kern(input_dim, [part])

def white(input_dim,variance=1.):
    """
     Construct a white kernel.

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float

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

    :param input_dim: dimensionality of the kernel, obligatory
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float

    """
    part = parts.bias.Bias(input_dim, variance)
    return kern(input_dim, [part])

def finite_dimensional(input_dim, F, G, variances=1., weights=None):
    """
    Construct a finite dimensional kernel.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param F: np.array of functions with shape (n,) - the n basis functions
    :type F: np.array
    :param G: np.array with shape (n,n) - the Gram matrix associated to F
    :type G: np.array
    :param variances: np.ndarray with shape (n,)
    :type: np.ndarray
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
    sympy_available = True
except ImportError:
    sympy_available = False

if sympy_available:
    from parts.sympykern import spkern
    from sympy.parsing.sympy_parser import parse_expr
    from GPy.util import symbolic

    def rbf_sympy(input_dim, ARD=False, variance=1., lengthscale=1.):
        """
        Radial Basis Function covariance.
        """
        X = sp.symbols('x_:' + str(input_dim))
        Z = sp.symbols('z_:' + str(input_dim))
        variance = sp.var('variance',positive=True)
        if ARD:
            lengthscales = sp.symbols('lengthscale_:' + str(input_dim))
            dist_string = ' + '.join(['(x_%i-z_%i)**2/lengthscale%i**2' % (i, i, i) for i in range(input_dim)])
            dist = parse_expr(dist_string)
            f =  variance*sp.exp(-dist/2.)
        else:
            lengthscale = sp.var('lengthscale',positive=True)
            dist_string = ' + '.join(['(x_%i-z_%i)**2' % (i, i) for i in range(input_dim)])
            dist = parse_expr(dist_string)
            f =  variance*sp.exp(-dist/(2*lengthscale**2))
        return kern(input_dim, [spkern(input_dim, f, name='rbf_sympy')])

    def eq_sympy(input_dim, output_dim, ARD=False):
        """
        Latent force model covariance, exponentiated quadratic with multiple outputs. Derived from a diffusion equation with the initial spatial condition layed down by a Gaussian process with lengthscale given by shared_lengthscale.

        See IEEE Trans Pattern Anal Mach Intell. 2013 Nov;35(11):2693-705. doi: 10.1109/TPAMI.2013.86. Linear latent force models using Gaussian processes. Alvarez MA, Luengo D, Lawrence ND.

        :param input_dim: Dimensionality of the kernel
        :type input_dim: int
        :param output_dim: number of outputs in the covariance function.
        :type output_dim: int
        :param ARD: whether or not to user ARD (default False).
        :type ARD: bool

        """
        real_input_dim = input_dim
        if output_dim>1:
            real_input_dim -= 1
        X = sp.symbols('x_:' + str(real_input_dim))
        Z = sp.symbols('z_:' + str(real_input_dim))
        scale = sp.var('scale_i scale_j',positive=True)
        if ARD:
            lengthscales = [sp.var('lengthscale%i_i lengthscale%i_j' % i, positive=True) for i in range(real_input_dim)]
            shared_lengthscales = [sp.var('shared_lengthscale%i' % i, positive=True) for i in range(real_input_dim)]
            dist_string = ' + '.join(['(x_%i-z_%i)**2/(shared_lengthscale%i**2 + lengthscale%i_i**2 + lengthscale%i_j**2)' % (i, i, i) for i in range(real_input_dim)])
            dist = parse_expr(dist_string)
            f =  variance*sp.exp(-dist/2.)
        else:
            lengthscales = sp.var('lengthscale_i lengthscale_j',positive=True)
            shared_lengthscale = sp.var('shared_lengthscale',positive=True)
            dist_string = ' + '.join(['(x_%i-z_%i)**2' % (i, i) for i in range(real_input_dim)])
            dist = parse_expr(dist_string)
            f =  scale_i*scale_j*sp.exp(-dist/(2*(lengthscale_i**2 + lengthscale_j**2 + shared_lengthscale**2)))
        return kern(input_dim, [spkern(input_dim, f, output_dim=output_dim, name='eq_sympy')])

    def ode1_eq(output_dim=1):
        """
        Latent force model covariance, first order differential
        equation driven by exponentiated quadratic.

        See N. D. Lawrence, G. Sanguinetti and M. Rattray. (2007)
        'Modelling transcriptional regulation using Gaussian
        processes' in B. Schoelkopf, J. C. Platt and T. Hofmann (eds)
        Advances in Neural Information Processing Systems, MIT Press,
        Cambridge, MA, pp 785--792.

        :param output_dim: number of outputs in the covariance function.
        :type output_dim: int
        """
        input_dim = 2
        x_0, z_0, decay_i, decay_j, scale_i, scale_j, lengthscale = sp.symbols('x_0, z_0, decay_i, decay_j, scale_i, scale_j, lengthscale')
        f = scale_i*scale_j*(symbolic.h(x_0, z_0, decay_i, decay_j, lengthscale) 
     + symbolic.h(z_0, x_0, decay_j, decay_i, lengthscale))
        return kern(input_dim, [spkern(input_dim, f, output_dim=output_dim, name='ode1_eq')])

    def sympykern(input_dim, k=None, output_dim=1, name=None, param=None):
        """
        A base kernel object, where all the hard work in done by sympy.

        :param k: the covariance function
        :type k: a positive definite sympy function of x1, z1, x2, z2...

        To construct a new sympy kernel, you'll need to define:
         - a kernel function using a sympy object. Ensure that the kernel is of the form k(x,z).
         - that's it! we'll extract the variables from the function k.

        Note:
         - to handle multiple inputs, call them x1, z1, etc
         - to handle multpile correlated outputs, you'll need to define each covariance function and 'cross' variance function. TODO
        """
        return kern(input_dim, [spkern(input_dim, k=k, output_dim=output_dim, name=name, param=param)])
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

    The symmetric kernel works by adding two GP functions together, and computing the overall covariance.

    Let f ~ GP(x | 0, k(x, x')). Now let g = f(x) + f(-x).

    It's easy to see that g is a symmetric function: g(x) = g(-x).

    by construction, g, is a gaussian Process with mean 0 and covariance

    k(x, x') + k(-x, x') + k(x, -x') + k(-x, -x')

    This constructor builds a covariance function of this form from the initial kernel
    """
    k_ = k.copy()
    k_.parts = [parts.symmetric.Symmetric(p) for p in k.parts]
    return k_

def coregionalize(output_dim,rank=1, W=None, kappa=None):
    """
    Coregionlization matrix B, of the form:

    .. math::
       \mathbf{B} = \mathbf{W}\mathbf{W}^\top + kappa \mathbf{I}

    An intrinsic/linear coregionalization kernel of the form:

    .. math::
       k_2(x, y)=\mathbf{B} k(x, y)

    it is obtainded as the tensor product between a kernel k(x,y) and B.

    :param output_dim: the number of outputs to corregionalize
    :type output_dim: int
    :param rank: number of columns of the W matrix (this parameter is ignored if parameter W is not None)
    :type rank: int
    :param W: a low rank matrix that determines the correlations between the different outputs, together with kappa it forms the coregionalization matrix B
    :type W: numpy array of dimensionality (num_outpus, rank)
    :param kappa: a vector which allows the outputs to behave independently
    :type kappa: numpy array of dimensionality  (output_dim,)
    :rtype: kernel object

    """
    p = parts.coregionalize.Coregionalize(output_dim,rank,W,kappa)
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
    TODO This can't be right! Construct a kernel with independent outputs from an existing kernel
    """
    # for sl in k.input_slices:
    #     assert (sl.start is None) and (sl.stop is None), "cannot adjust input slices! (TODO)"
    _parts = [parts.hierarchical.Hierarchical(k.parts)]
    return kern(k.input_dim+len(k.parts),_parts)

def build_lcm(input_dim, output_dim, kernel_list = [], rank=1,W=None,kappa=None):
    """
    Builds a kernel of a linear coregionalization model

    :input_dim: Input dimensionality
    :output_dim: Number of outputs
    :kernel_list: List of coregionalized kernels, each element in the list will be multiplied by a different corregionalization matrix
    :type kernel_list: list of GPy kernels
    :param rank: number tuples of the corregionalization parameters 'coregion_W'
    :type rank: integer

    ..note the kernels dimensionality is overwritten to fit input_dim

    """

    for k in kernel_list:
        if k.input_dim <> input_dim:
            k.input_dim = input_dim
            warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    k_coreg = coregionalize(output_dim,rank,W,kappa)
    kernel = kernel_list[0]**k_coreg.copy()

    for k in kernel_list[1:]:
        k_coreg = coregionalize(output_dim,rank,W,kappa)
        kernel += k**k_coreg.copy()

    return kernel

def ODE_1(input_dim=1, varianceU=1.,  varianceY=1., lengthscaleU=None,  lengthscaleY=None):
    """
    kernel resultiong from a first order ODE with OU driving GP

    :param input_dim: the number of input dimension, has to be equal to one
    :type input_dim: int
    :param varianceU: variance of the driving GP
    :type varianceU: float
    :param lengthscaleU: lengthscale of the driving GP
    :type lengthscaleU: float
    :param varianceY: 'variance' of the transfer function
    :type varianceY: float
    :param lengthscaleY: 'lengthscale' of the transfer function
    :type lengthscaleY: float
    :rtype: kernel object

    """
    part = parts.ODE_1.ODE_1(input_dim, varianceU, varianceY, lengthscaleU, lengthscaleY)
    return kern(input_dim, [part])

def ODE_UY(input_dim=2, varianceU=1.,  varianceY=1., lengthscaleU=None,  lengthscaleY=None):
    """
    kernel resultiong from a first order ODE with OU driving GP
    :param input_dim: the number of input dimension, has to be equal to one
    :type input_dim: int
    :param input_lengthU: the number of input U length
    :param varianceU: variance of the driving GP
    :type varianceU: float
    :param varianceY: 'variance' of the transfer function
    :type varianceY: float
    :param lengthscaleY: 'lengthscale' of the transfer function
    :type lengthscaleY: float
    :rtype: kernel object
    """
    part = parts.ODE_UY.ODE_UY(input_dim, varianceU, varianceY, lengthscaleU, lengthscaleY)
    return kern(input_dim, [part])
