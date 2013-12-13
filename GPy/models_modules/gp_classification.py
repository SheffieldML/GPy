# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPClassification(GP):
    """
    Gaussian Process classification

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values, can be None if likelihood is not None
    :param likelihood: a GPy likelihood, defaults to Bernoulli with Probit link_function
    :param kernel: a GPy kernel, defaults to rbf
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self,X,Y=None,likelihood=None,kernel=None,normalize_X=False,normalize_Y=False):
        if kernel is None:
            kernel = kern.rbf(X.shape[1])

        if likelihood is None:
            noise_model = likelihoods.bernoulli()
            likelihood = likelihoods.EP(Y, noise_model)
        elif Y is not None:
            if not all(Y.flatten() == likelihood.data.flatten()):
                raise Warning, 'likelihood.data and Y are different.'

        GP.__init__(self, X, likelihood, kernel, normalize_X=normalize_X)
        self.ensure_default_constraints()
