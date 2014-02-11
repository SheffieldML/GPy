# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from IPython.core.debugger import Tracer; debug_here=Tracer()
from kernpart import Kernpart
import numpy as np
from ...util.linalg import tdot
from ...core.mapping import Mapping
import GPy

class Hetero(Kernpart):
    """
    TODO: Need to constrain the function outputs
    positive (still thinking of best way of doing this!!! Yes, intend to use
    transformations, but what's the *best* way). Currently just squaring output.

    Heteroschedastic noise which depends on input location. See, for example,
    this paper by Goldberg et al.

    .. math::

       k(x_i, x_j) = \delta_{i,j} \sigma^2(x_i)

       where :math:`\sigma^2(x)` is a function giving the variance  as a function of input space and :math:`\delta_{i,j}` is the Kronecker delta function.

    The parameters are the parameters of \sigma^2(x) which is a
    function that can be specified by the user, by default an
    multi-layer peceptron is used.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param mapping: the mapping that gives the lengthscale across the input space (by default GPy.mappings.MLP is used with 20 hidden nodes).
    :type mapping: GPy.core.Mapping
    :rtype: Kernpart object

    See this paper:

    Goldberg, P. W.  Williams, C. K. I. and Bishop,
    C. M. (1998) Regression with Input-dependent Noise: a Gaussian
    Process Treatment In Advances in Neural Information Processing
    Systems, Volume 10, pp.  493-499. MIT Press

    for a Gaussian process treatment of this problem.

    """

    def __init__(self, input_dim, mapping=None, transform=None):
        self.input_dim = input_dim
        if not mapping:
            mapping = GPy.mappings.MLP(output_dim=1, hidden_dim=20, input_dim=input_dim)
        if not transform:
            transform = GPy.core.transformations.logexp()

        self.transform = transform
        self.mapping = mapping
        self.name='hetero'
        self.num_params=self.mapping.num_params
        self._set_params(self.mapping._get_params())

    def _get_params(self):
        return self.mapping._get_params()

    def _set_params(self, x):
        assert x.size == (self.num_params)
        self.mapping._set_params(x)

    def _get_param_names(self):
        return self.mapping._get_param_names()

    def K(self, X, X2, target):
        """Return covariance between X and X2."""
        if (X2 is None) or (X2 is X):
            target[np.diag_indices_from(target)] += self._Kdiag(X)

    def Kdiag(self, X, target):
        """Compute the diagonal of the covariance matrix for X."""
        target+=self._Kdiag(X)

    def _Kdiag(self, X):
        """Helper function for computing the diagonal elements of the covariance."""
        return self.mapping.f(X).flatten()**2

    def _param_grad_helper(self, dL_dK, X, X2, target):
        """Derivative of the covariance with respect to the parameters."""
        if (X2 is None) or (X2 is X):
            dL_dKdiag = dL_dK.flat[::dL_dK.shape[0]+1]
            self.dKdiag_dtheta(dL_dKdiag, X, target)

    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        """Gradient of diagonal of covariance with respect to parameters."""
        target += 2.*self.mapping.df_dtheta(dL_dKdiag[:, None]*self.mapping.f(X), X)

    def gradients_X(self, dL_dK, X, X2, target):
        """Derivative of the covariance matrix with respect to X."""
        if X2==None or X2 is X:
            dL_dKdiag = dL_dK.flat[::dL_dK.shape[0]+1]
            self.dKdiag_dX(dL_dKdiag, X, target)

    def dKdiag_dX(self, dL_dKdiag, X, target):
        """Gradient of diagonal of covariance with respect to X."""
        target += 2.*self.mapping.df_dX(dL_dKdiag[:, None], X)*self.mapping.f(X)



