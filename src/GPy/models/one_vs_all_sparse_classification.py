# Copyright (c) 2013, the GPy Authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy

class OneVsAllSparseClassification(object):
    """
    Gaussian Process classification: One vs all

    This is a thin wrapper around the models.GPClassification class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values, can be None if likelihood is not None
    :param kernel: a GPy kernel, defaults to rbf

    .. Note:: Multiple independent outputs are not allowed

    """

    def __init__(self, X, Y, kernel=None,Y_metadata=None,messages=True,num_inducing=10):
        if kernel is None:
            kernel = GPy.kern.RBF(X.shape[1]) + GPy.kern.White(X.shape[1]) + GPy.kern.Bias(X.shape[1])

        likelihood = GPy.likelihoods.Bernoulli()

        assert Y.shape[1] == 1, 'Y should be 1 column vector'

        labels = np.unique(Y.flatten())

        self.results = {}
        for yj in labels:
            print('Class %s vs all' %yj)
            Ynew = Y.copy()
            Ynew[Y.flatten()!=yj] = 0
            Ynew[Y.flatten()==yj] = 1

            m = GPy.models.SparseGPClassification(X,Ynew,kernel=kernel.copy(),Y_metadata=Y_metadata,num_inducing=num_inducing)
            m.optimize(messages=messages)
            self.results[yj] = m.predict(X)[0]
            del m
            del Ynew
