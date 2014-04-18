# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from ..likelihoods import likelihood
from ..inference.latent_function_inference import expectation_propagation_dtc

class SparseGPClassification(SparseGP):
    """
    sparse Gaussian Process model for classification

    This is a thin wrapper around the sparse_GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param likelihood: a GPy likelihood, defaults to Binomial with probit link_function
    :param kernel: a GPy kernel, defaults to rbf+white
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :rtype: model object

    """

    #def __init__(self, X, Y=None, likelihood=None, kernel=None, normalize_X=False, normalize_Y=False, Z=None, num_inducing=10):
    def __init__(self, X, Y=None, likelihood=None, kernel=None, Z=None, num_inducing=10, Y_metadata=None):


        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        if Z is None:
            i = np.random.permutation(X.shape[0])[:num_inducing]
            Z = X[i].copy()
        else:
            assert Z.shape[1] == X.shape[1]

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method=expectation_propagation_dtc.EPDTC(), name='SparseGPClassification',Y_metadata=Y_metadata)
    #def __init__(self, X, Y, Z, kernel, likelihood, inference_method=None, name='sparse gp', Y_metadata=None):
