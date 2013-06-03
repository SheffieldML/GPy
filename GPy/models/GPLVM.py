### Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
import sys, pdb
from .. import kern
from ..core import model
from ..util.linalg import pdinv, PCA
from GP import GP
from ..likelihoods import Gaussian
from .. import util
from GPy.util import plot_latent


class GPLVM(GP):
    """
    Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, Q, init='PCA', X = None, kernel=None, normalize_Y=False, **kwargs):
        if X is None:
            X = self.initialise_latent(init, Q, Y)
        if kernel is None:
            kernel = kern.rbf(Q, ARD=Q>1) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
        likelihood = Gaussian(Y, normalize=normalize_Y)
        GP.__init__(self, X, likelihood, kernel, **kwargs)

    def initialise_latent(self, init, Q, Y):
        if init == 'PCA':
            return PCA(Y, Q)[0]
        else:
            return np.random.randn(Y.shape[0], Q)

    def _get_param_names(self):
        return sum([['X_%i_%i'%(n,q) for q in range(self.Q)] for n in range(self.N)],[]) + GP._get_param_names(self)

    def _get_params(self):
        return np.hstack((self.X.flatten(), GP._get_params(self)))

    def _set_params(self,x):
        self.X = x[:self.N*self.Q].reshape(self.N,self.Q).copy()
        GP._set_params(self, x[self.X.size:])

    def _log_likelihood_gradients(self):
        dL_dX = 2.*self.kern.dK_dX(self.dL_dK,self.X)

        return np.hstack((dL_dX.flatten(),GP._log_likelihood_gradients(self)))

    def plot(self):
        assert self.likelihood.Y.shape[1]==2
        pb.scatter(self.likelihood.Y[:,0],self.likelihood.Y[:,1],40,self.X[:,0].copy(),linewidth=0,cmap=pb.cm.jet)
        Xnew = np.linspace(self.X.min(),self.X.max(),200)[:,None]
        mu, var, upper, lower = self.predict(Xnew)
        pb.plot(mu[:,0], mu[:,1],'k',linewidth=1.5)

    def plot_latent(self, *args, **kwargs):
        util.plot_latent.plot_latent(self, *args, **kwargs)