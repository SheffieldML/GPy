# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern

class SparseGPRegression(SparseGP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the SparseGP class, with a set of sensible defalts

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf+white
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param num_inducing: number of inducing points (ignored if Z is passed, see note)
    :type num_inducing: int
    :rtype: model object

    .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored
    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Z=None, num_inducing=10, X_variance=None):
        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.rbf(input_dim)  + kern.white(input_dim, variance=1e-3)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[:min(num_inducing, num_data)]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == input_dim

        likelihood = likelihoods.Gaussian()

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, X_variance=X_variance)
        self.ensure_default_constraints()

    def _getstate(self):
        return SparseGP._getstate(self)

    def _setstate(self, state):
        return SparseGP._setstate(self, state)



class SparseGPRegressionUncertainInput(SparseGP):
    """
    Gaussian Process model for regression with Gaussian variance on the inputs (X_variance)

    This is a thin wrapper around the SparseGP class, with a set of sensible defalts

    """

    def __init__(self, X, X_variance, Y, kernel=None, Z=None, num_inducing=10):
        """
        :param X: input observations
        :type X: np.ndarray (num_data x input_dim)
        :param X_variance: The uncertainty in the measurements of X (Gaussian variance, optional)
        :type X_variance: np.ndarray (num_data x input_dim)
        :param Y: observed values
        :param kernel: a GPy kernel, defaults to rbf+white
        :param Z: inducing inputs (optional, see note)
        :type Z: np.ndarray (num_inducing x input_dim) | None
        :param num_inducing: number of inducing points (ignored if Z is passed, see note)
        :type num_inducing: int
        :rtype: model object

        .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored
        .. Note:: Multiple independent outputs are allowed using columns of Y
        """
        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.rbf(input_dim)  + kern.white(input_dim, variance=1e-3)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[:min(num_inducing, num_data)]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == input_dim

        likelihood = likelihoods.Gaussian()

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, X_variance=X_variance)
        self.ensure_default_constraints()
