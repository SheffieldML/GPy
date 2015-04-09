# Copyright (c) 2014, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
class Poly(Kern):
    """
    Polynomial kernel
    """

    def __init__(self, input_dim, variance=1., order=3., active_dims=None, name='poly'):
        super(Poly, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.link_parameter(self.variance)
        self.order=order

    def K(self, X, X2=None):
        return (self._dot_product(X, X2) + 1.)**self.order * self.variance

    def _dot_product(self, X, X2=None):
        if X2 is None:
            return np.dot(X, X.T)
        else:
            return np.dot(X, X2.T)

    def Kdiag(self, X):
        return self.variance*(np.square(X).sum(1) + 1.)**self.order

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.sum(dL_dK * (self._dot_product(X, X2) + 1.)**self.order)

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError
