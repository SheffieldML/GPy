# Copyright (c) 2014, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

class Poly(Kern):
    """
    Polynomial kernel
    """

    def __init__(self, input_dim, variance=1., scale=1., bias=1., order=3., active_dims=None, name='poly'):
        super(Poly, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.scale = Param('scale', scale, Logexp())
        self.bias = Param('bias', bias, Logexp())

        self.link_parameters(self.variance, self.scale, self.bias)
        assert order >= 1, 'The order of the polynomial has to be at least 1.'
        self.order=order


    def K(self, X, X2=None):
        _, _, B = self._AB(X, X2)
        return B * self.variance

    @Cache_this(limit=3)
    def _AB(self, X, X2=None):
        if X2 is None:
            dot_prod = np.dot(X, X.T)
        else:
            dot_prod = np.dot(X, X2.T)
        A = (self.scale * dot_prod) + self.bias
        B = A ** self.order
        return dot_prod, A, B

    def Kdiag(self, X):
        return self.K(X).diagonal()#self.variance*(np.square(X).sum(1) + 1.)**self.order

    def update_gradients_full(self, dL_dK, X, X2=None):
        dot_prod, A, B = self._AB(X, X2)
        dK_dA = self.variance * self.order * A ** (self.order-1.)
        dL_dA = dL_dK * (dK_dA)
        self.scale.gradient = (dL_dA * dot_prod).sum()
        self.bias.gradient = dL_dA.sum()
        self.variance.gradient = np.sum(dL_dK * B)
        #import ipdb;ipdb.set_trace()

    def update_gradients_diag(self, dL_dKdiag, X):
        raise NotImplementedError

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def gradients_X_diag(self, dL_dKdiag, X):
        raise NotImplementedError

# I haven't tested the following functions out nor am I sure about them.
# Will come back to this at a later stage

#    def update_gradients_diag(self, dL_dKdiag, X):
#        dot_prod = (x**2).sum()
#         A = (self.scale * dot_prod) + self.bias
#         B = A ** self.order
#         dK_dA = self.variance * self.order * A ** (self.order-1.)
#         dL_dA = dL_dKdiag * (dK_dA)
#         self.scale.gradient = (dL_dA * dot_prod).sum()
#         self.bias.gradient = dL_dA.sum()
#         self.variance.gradient = np.sum(dL_dKdiag * B)
#
#     def gradients_X(self, dL_dK, X, X2=None):
#         dot_prod, A, _ = self._AB(X, X2)
#         dK_dA = self.variance * self.order * A ** (self.order-1.)
#         if X2 is None:
#             return dL_dK * (dK_dA*self.scale*2*X)
#         else:
#             return dL_dK * (dK_dA*self.scale*X)
#
#     def gradients_X_diag(self, dL_dKdiag, X):
#         dot_prod = (x**2).sum()
#         A = (self.scale * dot_prod) + self.bias
#         dK_dA = self.variance * self.order * A ** (self.order-1.)
#         return dL_dKdiag * (dK_dA*self.scale*2*X)
