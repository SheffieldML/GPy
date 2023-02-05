# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import numpy as np


class WienerVelocity(Kern):
    """
    Wiener Velocity is a 1D kernel only.
    Negative times are treated as a separate (backwards!) motion.
    The Wiener velocity kernel corresponds to a once integrated Brownian motion kernel,
    as described in Solin: "Stochastic Differential Equation Methods for Spatio-Temporal Gaussian Process Regression", 2016.
        URL: http://urn.fi/URN:ISBN:978-952-60-6711-7

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """

    def __init__(self, input_dim=1, variance=1., active_dims=None, name='WienerVelocity'):
        assert input_dim == 1, "Wiener velocity in 1D only"
        super(WienerVelocity, self).__init__(input_dim, active_dims, name)

        self.variance = Param('variance', variance, Logexp())
        self.link_parameters(self.variance)

    def to_dict(self):
        input_dict = super(WienerVelocity, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.WienerVelocity"
        input_dict["variance"] = self.variance.values.tolist()
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return WienerVelocity(**input_dict)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return (self.variance*np.where(np.sign(X) == np.sign(X2.T), (np.fmin(np.abs(X), np.abs(X2.T))**3) /
                                       3 + np.abs(X - X2.T) * (np.fmin(np.abs(X), np.abs(X2.T))**2) / 2, 0.))

    def Kdiag(self, X):
        return self.variance*np.divide(np.abs(X.flatten())**3, 3)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        self.variance.gradient = (np.sum(dL_dK * np.where(np.sign(X) == np.sign(X2.T), (np.fmin(np.abs(X), np.abs(X2.T))**3) /
                                                          3 + np.abs(X - X2.T) * (np.fmin(np.abs(X), np.abs(X2.T))**2) / 2, 0.)))
