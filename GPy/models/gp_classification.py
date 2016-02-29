# Copyright (c) 2013, the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import GP
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference.expectation_propagation import EP

class GPClassification(GP):
    """
    Gaussian Process classification

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values, can be None if likelihood is not None
    :param kernel: a GPy kernel, defaults to rbf

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None,Y_metadata=None, mean_function=None):
        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        likelihood = likelihoods.Bernoulli()

        GP.__init__(self, X=X, Y=Y,  kernel=kernel, likelihood=likelihood, inference_method=EP(), mean_function=mean_function, name='gp_classification')
