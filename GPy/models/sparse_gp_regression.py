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
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, normalize_X=False, normalize_Y=False, Z=None, num_inducing=10, X_variance=None):
        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.rbf(X.shape[1]) + kern.white(X.shape[1], 1e-3)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        # likelihood defaults to Gaussian
        likelihood = likelihoods.Gaussian(Y, normalize=normalize_Y)

        SparseGP.__init__(self, X, likelihood, kernel, Z=Z, normalize_X=normalize_X, X_variance=X_variance)
        self._set_params(self._get_params())
