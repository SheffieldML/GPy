# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
import sys, pdb
from .. import kern
from ..core import model
from ..util.linalg import pdinv, PCA
from GP_regression import GP_regression

class GPLVM(GP_regression):
    """
    Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, Q, init='PCA', X = None, **kwargs):
        if X is None:
            X = self.initialise_latent(init, Q, Y)
        GP_regression.__init__(self, X, Y, **kwargs)

    def initialise_latent(self, init, Q, Y):
        if init == 'PCA':
            return PCA(Y, Q)[0]
        else:
            return np.random.randn(Y.shape[0], Q)

    def get_param_names(self):
        return (sum([['X_%i_%i'%(n,q) for n in range(self.N)] for q in range(self.Q)],[])
                + self.kern.extract_param_names())

    def get_param(self):
        return np.hstack((self.X.flatten(), self.kern.extract_param()))

    def set_param(self,x):
        self.X = x[:self.X.size].reshape(self.N,self.Q).copy()
        GP_regression.set_param(self, x[self.X.size:])

    def log_likelihood_gradients(self):
        dL_dK = self.dL_dK()

        dL_dtheta = self.kern.dK_dtheta(dL_dK,self.X)
        dL_dX = 2*self.kern.dK_dX(dL_dK,self.X)

        return np.hstack((dL_dX.flatten(),dL_dtheta))

    def plot(self):
        assert self.Y.shape[1]==2
        pb.scatter(self.Y[:,0],self.Y[:,1],40,self.X[:,0].copy(),linewidth=0)
        Xnew = np.linspace(self.X.min(),self.X.max(),200)[:,None]
        mu, var = self.predict(Xnew)
        pb.plot(mu[:,0], mu[:,1],'k',linewidth=1.5)

    def plot_latent(self):
        raise NotImplementedError
