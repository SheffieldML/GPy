# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern
from .. import util

class GPHeteroscedasticRegression(GP):
    """
    Gaussian Process model for heteroscedastic regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf

    NB: This model does not make inference on the noise outside the training set
    """
    def __init__(self, X, Y, kernel=None, Y_metadata=None):

        Ny = Y.shape[0]

        if Y_metadata is None:
            Y_metadata = {'output_index':np.arange(Ny)[:,None]}
        else:
            assert Y_metadata['output_index'].shape[0] == Ny

        if kernel is None:
            kernel = kern.RBF(X.shape[1])

        #Likelihood
        likelihood = likelihoods.HeteroscedasticGaussian(Y_metadata)

        super(GPHeteroscedasticRegression, self).__init__(X,Y,kernel,likelihood, Y_metadata=Y_metadata)

