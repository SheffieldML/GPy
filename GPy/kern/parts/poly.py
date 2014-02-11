# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
four_over_tau = 2./np.pi

class POLY(Kernpart):
    """

    Polynomial kernel parameter initialisation.  Included for completeness, but generally not recommended, is the polynomial kernel:

    .. math::
        k(x, y) = \sigma^2\*(\sigma_w^2 x'y+\sigma_b^b)^d

    The kernel parameters are :math:`\sigma^2` (variance), :math:`\sigma^2_w`
    (weight_variance), :math:`\sigma^2_b` (bias_variance) and d
    (degree). Only gradients of the first three are provided for
    kernel optimisation, it is assumed that polynomial degree would
    be set by hand.

    The kernel is not recommended as it is badly behaved when the
    :math:`\sigma^2_w\*x'\*y + \sigma^2_b` has a magnitude greater than one. For completeness
    there is an automatic relevance determination version of this
    kernel provided (NOTE YET IMPLEMENTED!).
    :param input_dim: the number of input dimensions
    :type input_dim: int 
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param weight_variance: the vector of the variances of the prior over input weights in the neural network :math:`\sigma^2_w`
    :type weight_variance: array or list of the appropriate size (or float if there is only one weight variance parameter)
    :param bias_variance: the variance of the prior over bias parameters :math:`\sigma^2_b`
    :param degree: the degree of the polynomial.
    :type degree: int
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one weight variance parameter :math:`\sigma^2_w`), otherwise there is one weight variance parameter per dimension.
    :type ARD: Boolean
    :rtype: Kernpart object

    """

    def __init__(self, input_dim, variance=1., weight_variance=None, bias_variance=1., degree=2, ARD=False):
        self.input_dim = input_dim
        self.ARD = ARD
        if not ARD:
            self.num_params=3
            if weight_variance is not None:
                weight_variance = np.asarray(weight_variance)
                assert weight_variance.size == 1, "Only one weight variance needed for non-ARD kernel"
            else:
                weight_variance = 1.*np.ones(1)
        else:
            self.num_params = self.input_dim + 2
            if weight_variance is not None:
                weight_variance = np.asarray(weight_variance)
                assert weight_variance.size == self.input_dim, "bad number of weight variances"
            else:
                weight_variance = np.ones(self.input_dim)
            raise NotImplementedError
        self.degree=degree
        self.name='poly_deg' + str(self.degree)
        self._set_params(np.hstack((variance, weight_variance.flatten(), bias_variance)))

    def _get_params(self):
        return np.hstack((self.variance, self.weight_variance.flatten(), self.bias_variance))

    def _set_params(self, x):
        assert x.size == (self.num_params)
        self.variance = x[0]
        self.weight_variance = x[1:-1]
        self.weight_std = np.sqrt(self.weight_variance)
        self.bias_variance = x[-1]

    def _get_param_names(self):
        if self.num_params == 3:
            return ['variance', 'weight_variance', 'bias_variance']
        else:
            return ['variance'] + ['weight_variance_%i' % i for i in range(self.lengthscale.size)] + ['bias_variance']

    def K(self, X, X2, target):
        """Return covariance between X and X2."""
        self._K_computations(X, X2)
        target += self.variance*self._K_dvar

    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix for X."""
        self._K_diag_computations(X)
        target+= self.variance*self._K_diag_dvar

    def _param_grad_helper(self, dL_dK, X, X2, target):
        """Derivative of the covariance with respect to the parameters."""
        self._K_computations(X, X2)
        base = self.variance*self.degree*self._K_poly_arg**(self.degree-1)
        base_cov_grad = base*dL_dK


            
        target[0] += np.sum(self._K_dvar*dL_dK)
        target[1] += (self._K_inner_prod*base_cov_grad).sum()
        target[2] += base_cov_grad.sum()


    def gradients_X(self, dL_dK, X, X2, target):
        """Derivative of the covariance matrix with respect to X"""
        self._K_computations(X, X2)
        arg = self._K_poly_arg
        if X2 is None:
            target += 2*self.weight_variance*self.degree*self.variance*(((X[None,:, :])) *(arg**(self.degree-1))[:, :, None]*dL_dK[:, :, None]).sum(1)
        else:
            target += self.weight_variance*self.degree*self.variance*(((X2[None,:, :])) *(arg**(self.degree-1))[:, :, None]*dL_dK[:, :, None]).sum(1)
            
    def dKdiag_dX(self, dL_dKdiag, X, target):
        """Gradient of diagonal of covariance with respect to X"""
        self._K_diag_computations(X)
        arg = self._K_diag_poly_arg
        target += 2.*self.weight_variance*self.degree*self.variance*X*dL_dKdiag[:, None]*(arg**(self.degree-1))[:, None]
    
    
    def _K_computations(self, X, X2):
        if self.ARD:
            pass
        else:
            if X2 is None:
                self._K_inner_prod = np.dot(X,X.T)
            else:
                self._K_inner_prod = np.dot(X,X2.T)
            self._K_poly_arg = self._K_inner_prod*self.weight_variance + self.bias_variance
        self._K_dvar = self._K_poly_arg**self.degree

    def _K_diag_computations(self, X):
        if self.ARD:
            pass
        else:
            self._K_diag_poly_arg = (X*X).sum(1)*self.weight_variance + self.bias_variance
        self._K_diag_dvar = self._K_diag_poly_arg**self.degree

  


