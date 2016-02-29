# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np

class Brownian(Kern):
    """
    Brownian motion in 1D only.

    Negative times are treated as a separate (backwards!) Brownian motion.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """
    def __init__(self, input_dim=1, variance=1., active_dims=None, name='Brownian'):
        assert input_dim==1, "Brownian motion in 1D only"
        super(Brownian, self).__init__(input_dim, active_dims, name)

        self.variance = Param('variance', variance, Logexp())
        self.link_parameters(self.variance)

    def K(self,X,X2=None):
        if X2 is None:
            X2 = X
        return self.variance*np.where(np.sign(X)==np.sign(X2.T),np.fmin(np.abs(X),np.abs(X2.T)), 0.)

    def Kdiag(self,X):
        return self.variance*np.abs(X.flatten())

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        self.variance.gradient = np.sum(dL_dK * np.where(np.sign(X)==np.sign(X2.T),np.fmin(np.abs(X),np.abs(X2.T)), 0.))

    #def update_gradients_diag(self, dL_dKdiag, X):
        #self.variance.gradient = np.dot(np.abs(X.flatten()), dL_dKdiag)

    #def gradients_X(self, dL_dK, X, X2=None):
        #if X2 is None:
            #return np.sum(self.variance*dL_dK*np.abs(X),1)[:,None]
        #else:
            #return np.sum(np.where(np.logical_and(np.abs(X)<np.abs(X2.T), np.sign(X)==np.sign(X2)), self.variance*dL_dK,0.),1)[:,None]



