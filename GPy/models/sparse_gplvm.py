# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
import sys, pdb
from GPy.models.sparse_gp_regression import SparseGPRegression
from GPy.models.gplvm import GPLVM
# from .. import kern
# from ..core import model
# from ..util.linalg import pdinv, PCA

class SparseGPLVM(SparseGPRegression, GPLVM):
    """
    Sparse Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, kernel=None, init='PCA', num_inducing=10):
        X = self.initialise_latent(init, input_dim, Y)
        SparseGPRegression.__init__(self, X, Y, kernel=kernel, num_inducing=num_inducing)

    def _get_param_names(self):
        return (sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
                + SparseGPRegression._get_param_names(self))

    def _get_params(self):
        return np.hstack((self.X.flatten(), SparseGPRegression._get_params(self)))

    def _set_params(self, x):
        self.X = x[:self.X.size].reshape(self.num_data, self.input_dim).copy()
        SparseGPRegression._set_params(self, x[self.X.size:])

    def log_likelihood(self):
        return SparseGPRegression.log_likelihood(self)

    def dL_dX(self):
        dL_dX = self.kern.dKdiag_dX(self.dL_dpsi0, self.X)
        dL_dX += self.kern.dK_dX(self.dL_dpsi1, self.X, self.Z)

        return dL_dX

    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dX().flatten(), SparseGPRegression._log_likelihood_gradients(self)))

    def plot(self):
        GPLVM.plot(self)
        # passing Z without a small amout of jitter will induce the white kernel where we don;t want it!
        mu, var, upper, lower = SparseGPRegression.predict(self, self.Z + np.random.randn(*self.Z.shape) * 0.0001)
        pb.plot(mu[:, 0] , mu[:, 1], 'ko')

    def plot_latent(self, *args, **kwargs):
        input_1, input_2 = GPLVM.plot_latent(*args, **kwargs)
        pb.plot(m.Z[:, input_1], m.Z[:, input_2], '^w')
