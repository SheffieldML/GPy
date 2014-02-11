# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
from ...util.linalg import tdot
from ...core.mapping import Mapping
import GPy

class Gibbs(Kernpart):
    """
    Gibbs non-stationary covariance function. 

    .. math::
       
       r = sqrt((x_i - x_j)'*(x_i - x_j))
       
       k(x_i, x_j) = \sigma^2*Z*exp(-r^2/(l(x)*l(x) + l(x')*l(x')))

       Z = (2*l(x)*l(x')/(l(x)*l(x) + l(x')*l(x')^{q/2}

       where :math:`l(x)` is a function giving the length scale as a function of space and :math:`q` is the dimensionality of the input space.
       This is the non stationary kernel proposed by Mark Gibbs in his 1997
        thesis. It is similar to an RBF but has a length scale that varies
        with input location. This leads to an additional term in front of
        the kernel.

        The parameters are :math:`\sigma^2`, the process variance, and
        the parameters of l(x) which is a function that can be
        specified by the user, by default an multi-layer peceptron is
        used.

        :param input_dim: the number of input dimensions
        :type input_dim: int 
        :param variance: the variance :math:`\sigma^2`
        :type variance: float
        :param mapping: the mapping that gives the lengthscale across the input space (by default GPy.mappings.MLP is used with 20 hidden nodes).
        :type mapping: GPy.core.Mapping
        :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one weight variance parameter \sigma^2_w), otherwise there is one weight variance parameter per dimension.
        :type ARD: Boolean
        :rtype: Kernpart object

    See Mark Gibbs's thesis for more details: Gibbs,
    M. N. (1997). Bayesian Gaussian Processes for Regression and
    Classification. PhD thesis, Department of Physics, University of
    Cambridge. Or also see Page 93 of Gaussian Processes for Machine
    Learning by Rasmussen and Williams. Although note that we do not
    constrain the lengthscale to be positive by default. This allows
    anticorrelation to occur. The positive constraint can be included
    by the user manually.

    """

    def __init__(self, input_dim, variance=1., mapping=None, ARD=False):
        self.input_dim = input_dim
        self.ARD = ARD
        if not mapping:
            mapping = GPy.mappings.MLP(output_dim=1, hidden_dim=20, input_dim=input_dim)
        if not ARD:
            self.num_params=1+mapping.num_params
        else:
            raise NotImplementedError

        self.mapping = mapping
        self.name='gibbs'
        self._set_params(np.hstack((variance, self.mapping._get_params())))

    def _get_params(self):
        return np.hstack((self.variance, self.mapping._get_params()))

    def _set_params(self, x):
        assert x.size == (self.num_params)
        self.variance = x[0]
        self.mapping._set_params(x[1:])

    def _get_param_names(self):
        return ['variance'] + self.mapping._get_param_names()

    def K(self, X, X2, target):
        """Return covariance between X and X2."""
        self._K_computations(X, X2)
        target += self.variance*self._K_dvar

    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix for X."""
        np.add(target, self.variance, target)

    def _param_grad_helper(self, dL_dK, X, X2, target):
        """Derivative of the covariance with respect to the parameters."""
        self._K_computations(X, X2)
        self._dK_computations(dL_dK)
        if X2==None:
            gmapping = self.mapping.df_dtheta(2*self._dL_dl[:, None], X)
        else:
            gmapping = self.mapping.df_dtheta(self._dL_dl[:, None], X)
            gmapping += self.mapping.df_dtheta(self._dL_dl_two[:, None], X2)

        target+= np.hstack([(dL_dK*self._K_dvar).sum(), gmapping])

    def gradients_X(self, dL_dK, X, X2, target):
        """Derivative of the covariance matrix with respect to X."""
        # First account for gradients arising from presence of X in exponent.
        self._K_computations(X, X2)
        if X2 is None:
            _K_dist = 2*(X[:, None, :] - X[None, :, :])
        else:
            _K_dist = X[:, None, :] - X2[None, :, :] # don't cache this in _K_co
        gradients_X = (-2.*self.variance)*np.transpose((self._K_dvar/self._w2)[:, :, None]*_K_dist, (1, 0, 2))
        target += np.sum(gradients_X*dL_dK.T[:, :, None], 0)
        # Now account for gradients arising from presence of X in lengthscale.
        self._dK_computations(dL_dK)
        if X2 is None:
            target += 2.*self.mapping.df_dX(self._dL_dl[:, None], X)
        else:
            target += self.mapping.df_dX(self._dL_dl[:, None], X)
    
    def dKdiag_dX(self, dL_dKdiag, X, target):
        """Gradient of diagonal of covariance with respect to X."""
        pass

    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        """Gradient of diagonal of covariance with respect to parameters."""
        target[0] += np.sum(dL_dKdiag)


    
    def _K_computations(self, X, X2=None):
        """Pre-computations for the covariance function (used both when computing the covariance and its gradients). Here self._dK_dvar and self._K_dist2 are updated."""
        self._lengthscales=self.mapping.f(X)
        self._lengthscales2=np.square(self._lengthscales)
        if X2==None:
            self._lengthscales_two = self._lengthscales
            self._lengthscales_two2 = self._lengthscales2
            Xsquare = np.square(X).sum(1)
            self._K_dist2 = -2.*tdot(X) + Xsquare[:, None] + Xsquare[None, :]
        else:
            self._lengthscales_two = self.mapping.f(X2)
            self._lengthscales_two2 = np.square(self._lengthscales_two)
            self._K_dist2 = -2.*np.dot(X, X2.T) + np.square(X).sum(1)[:, None] + np.square(X2).sum(1)[None, :]
        self._w2 = self._lengthscales2 + self._lengthscales_two2.T
        prod_length = self._lengthscales*self._lengthscales_two.T
        self._K_exponential = np.exp(-self._K_dist2/self._w2)
        self._K_dvar = np.sign(prod_length)*(2*np.abs(prod_length)/self._w2)**(self.input_dim/2.)*np.exp(-self._K_dist2/self._w2)

    def _dK_computations(self, dL_dK):
        """Pre-computations for the gradients of the covaraince function. Here the gradient of the covariance with respect to all the individual lengthscales is computed.
        :param dL_dK: the gradient of the objective with respect to the covariance function.
        :type dL_dK: ndarray"""
        
        self._dL_dl = (dL_dK*self.variance*self._K_dvar*(self.input_dim/2.*(self._lengthscales_two.T**4 - self._lengthscales**4) + 2*self._lengthscales2*self._K_dist2)/(self._w2*self._w2*self._lengthscales)).sum(1)
        if self._lengthscales_two is self._lengthscales:
            self._dL_dl_two = None
        else:
            self._dL_dl_two = (dL_dK*self.variance*self._K_dvar*(self.input_dim/2.*(self._lengthscales**4 - self._lengthscales_two.T**4 ) + 2*self._lengthscales_two2.T*self._K_dist2)/(self._w2*self._w2*self._lengthscales_two.T)).sum(0)
