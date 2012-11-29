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

    def get_param_names(self):
        return (sum([['X_%i_%i'%(n,q) for n in range(self.N)] for q in range(self.Q)],[])
                + sparse_GP_regression.get_param_names(self))

    def get_param(self):
        return np.hstack((self.X.flatten(), sparse_GP_regression.get_param(self)))

    def set_param(self,x):
        self.X = x[:self.X.size].reshape(self.N,self.Q).copy()
        sparse_GP_regression.set_param(self, x[self.X.size:])

    def log_likelihood(self):
        return sparse_GP_regression.log_likelihood(self)

    def dL_dX(self):
        dpsi0_dX = self.kern.dKdiag_dX(self.X)
        dpsi1_dX = self.kern.dK_dX(self.X,self.Z)
        dpsi2_dX = self.psi1[:,None,:,None]*dpsi1_dX[None,:,:,:]

        dL_dX = ((self.dL_dpsi0 * dpsi0_dX).sum(0)
                 + (self.dL_dpsi1[:,:,None]*dpsi1_dX).sum(0)
                 + 2.0*(self.dL_dpsi2[:, :, None,None] * dpsi2_dX).sum(0).sum(0))

        return dL_dX

    def log_likelihood_gradients(self):
        return np.hstack((self.dL_dX().flatten(), sparse_GP_regression.log_likelihood_gradients(self)))

    def plot(self):
        GPLVM.plot(self)
        mu, var = sparse_GP_regression.predict(self, self.Z+np.random.randn(*self.Z.shape)*0.0001)
        pb.plot(mu[:, 0] , mu[:, 1], 'ko')
