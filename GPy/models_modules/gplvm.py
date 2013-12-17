# ## Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from .. import kern
from ..core import priors
from ..core import GP
from ..likelihoods import Gaussian
from .. import util
from ..util.linalg import pca

def initialise_latent(init, input_dim, Y):
    Xr = np.random.randn(Y.shape[0], input_dim)
    if init.lower() == 'pca':
        PC = pca(Y, input_dim)[0]
        Xr[:PC.shape[0], :PC.shape[1]] = PC
    return Xr

class GPLVM(GP):
    """
    Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'pca'|'random'

    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None, normalize_Y=False):
        if X is None:
            X = initialise_latent(init, input_dim, Y)
        if kernel is None:
            kernel = kern.rbf(input_dim, ARD=input_dim > 1) + kern.bias(input_dim, np.exp(-2))
        likelihood = Gaussian(Y, normalize=normalize_Y, variance=np.exp(-2.))
        GP.__init__(self, X, likelihood, kernel, normalize_X=False)
        self.set_prior('.*X', priors.Gaussian(0, 1))
        self.ensure_default_constraints()

    def _get_param_names(self):
        return sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], []) + GP._get_param_names(self)

    def _get_params(self):
        return np.hstack((self.X.flatten(), GP._get_params(self)))

    def _set_params(self, x):
        self.X = x[:self.num_data * self.input_dim].reshape(self.num_data, self.input_dim).copy()
        GP._set_params(self, x[self.X.size:])

    def _log_likelihood_gradients(self):
        dL_dX = self.kern.dK_dX(self.dL_dK, self.X)

        return np.hstack((dL_dX.flatten(), GP._log_likelihood_gradients(self)))

    def jacobian(self,X):
        target = np.zeros((X.shape[0],X.shape[1],self.output_dim))
        for i in range(self.output_dim):
            target[:,:,i] = self.kern.dK_dX(np.dot(self.Ki,self.likelihood.Y[:,i])[None, :],X,self.X)
        return target

    def magnification(self,X):
        target=np.zeros(X.shape[0])
        J = np.zeros((X.shape[0],X.shape[1],self.output_dim))
        J=self.jacobian(X)
        for i in range(X.shape[0]):
            target[i]=np.sqrt(pb.det(np.dot(J[i,:,:],np.transpose(J[i,:,:]))))
        return target

    def plot(self):
        assert self.likelihood.Y.shape[1] == 2
        pb.scatter(self.likelihood.Y[:, 0], self.likelihood.Y[:, 1], 40, self.X[:, 0].copy(), linewidth=0, cmap=pb.cm.jet)
        Xnew = np.linspace(self.X.min(), self.X.max(), 200)[:, None]
        mu, var, upper, lower = self.predict(Xnew)
        pb.plot(mu[:, 0], mu[:, 1], 'k', linewidth=1.5)

    def plot_latent(self, *args, **kwargs):
        return util.plot_latent.plot_latent(self, *args, **kwargs)

    def plot_magnification(self, *args, **kwargs):
        return util.plot_latent.plot_magnification(self, *args, **kwargs)

    def getstate(self):
        return GP.getstate(self)

    def setstate(self, state):
        GP.setstate(self, state)


