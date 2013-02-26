# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
import sys, pdb
# from .. import kern
# from ..core import model
# from ..util.linalg import pdinv, PCA
from GPLVM import GPLVM
from sparse_GP_regression import sparse_GP_regression

class sparse_GPLVM(sparse_GP_regression, GPLVM):
    """
    Sparse Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, Q, init='PCA', **kwargs):
        X = self.initialise_latent(init, Q, Y)
        sparse_GP_regression.__init__(self, X, Y, **kwargs)

    def _get_param_names(self):
        return (sum([['X_%i_%i'%(n,q) for q in range(self.Q)] for n in range(self.N)],[])
                + sparse_GP_regression._get_param_names(self))

    def _get_params(self):
        return np.hstack((self.X.flatten(), sparse_GP_regression._get_params(self)))

    def _set_params(self,x):
        self.X = x[:self.X.size].reshape(self.N,self.Q).copy()
        sparse_GP_regression._set_params(self, x[self.X.size:])

    def log_likelihood(self):
        return sparse_GP_regression.log_likelihood(self)

    def dL_dX(self):
        dL_dX = self.kern.dKdiag_dX(self.dL_dpsi0,self.X)
        dL_dX += self.kern.dK_dX(self.dL_dpsi1,self.X,self.Z)

        return dL_dX

    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dX().flatten(), sparse_GP_regression._log_likelihood_gradients(self)))

    def plot(self):
        GPLVM.plot(self)
        #passing Z without a small amout of jitter will induce the white kernel where we don;t want it!
        mu, var, upper, lower = sparse_GP_regression.predict(self, self.Z+np.random.randn(*self.Z.shape)*0.0001)
        pb.plot(mu[:, 0] , mu[:, 1], 'ko')
