# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPRegression(GP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None):

        if kernel is None:
            kernel = kern.rbf(X.shape[1])

        likelihood = likelihoods.Gaussian()

        super(GPRegression, self).__init__(X, Y, kernel, likelihood, name='gp_regression')

    def _getstate(self):
        return GP._getstate(self)

    def _setstate(self, state):
        return GP._setstate(self, state)
