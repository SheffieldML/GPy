# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from sparse_GP import sparse_GP
from .. import likelihoods
from .. import kern
from ..likelihoods import likelihood
from GP_regression import GP_regression

class sparse_GP_regression(sparse_GP):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the GP class, with a set of sensible defalts

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf+white
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :param Xslices: how the X,Y data co-vary in the kernel (i.e. which "outputs" they correspond to). See (link:slicing)
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self,X,Y,kernel=None,normalize_X=False,normalize_Y=False, Xslices=None,Z=None, M=10):
        #kern defaults to rbf
        if kernel is None:
            kernel = kern.rbf(X.shape[1]) + kern.white(X.shape[1],1e-3)

        #Z defaults to a subset of the data
        if Z is None:
            Z = np.random.permutation(X.copy())[:M]
        else:
            assert Z.shape[1]==X.shape[1]

        #likelihood defaults to Gaussian
        likelihood = likelihoods.Gaussian(Y,normalize=normalize_Y)

        sparse_GP.__init__(self, X, likelihood, kernel, Z, normalize_X=normalize_X, Xslices=Xslices)
