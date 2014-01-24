# ## Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from .. import kern
from ..util.linalg import PCA
from ..core import GP, Param
from ..likelihoods import Gaussian
from .. import util


class GPLVM(GP):
    """
    Gaussian Process Latent Variable Model


    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None, normalize_Y=False, name="gplvm"):

        """
        :param Y: observed data
        :type Y: np.ndarray
        :param input_dim: latent dimensionality
        :type input_dim: int
        :param init: initialisation method for the latent space
        :type init: 'PCA'|'random'
        """
        if X is None:
            X = self.initialise_latent(init, input_dim, Y)
        if kernel is None:
            kernel = kern.rbf(input_dim, ARD=input_dim > 1) + kern.bias(input_dim, np.exp(-2))

        likelihood = Gaussian()

        super(GPLVM, self).__init__(X, Y, kernel, likelihood, name='GPLVM')
        self.X = Param('X', X)
        self.add_parameter(self.X, index=0)

    def initialise_latent(self, init, input_dim, Y):
        Xr = np.random.randn(Y.shape[0], input_dim)
        if init == 'PCA':
            PC = PCA(Y, input_dim)[0]
            Xr[:PC.shape[0], :PC.shape[1]] = PC
        else:
            raise NotImplementedError
        return Xr

    def parameters_changed(self):
        GP.parameters_changed(self)
        self.X.gradient = self.kern.gradients_X(self.posterior.dL_dK, self.X)

    def _getstate(self):
        return GP._getstate(self)

    def _setstate(self, state):
        GP._setstate(self, state)

    def jacobian(self,X):
        target = np.zeros((X.shape[0],X.shape[1],self.output_dim))
        for i in range(self.output_dim):
            target[:,:,i]=self.kern.dK_dX(np.dot(self.Ki,self.likelihood.Y[:,i])[None, :],X,self.X)
        return target

    def magnification(self,X):
        target=np.zeros(X.shape[0])
        #J = np.zeros((X.shape[0],X.shape[1],self.output_dim))
        J = self.jacobian(X)
        for i in range(X.shape[0]):
            target[i]=np.sqrt(pb.det(np.dot(J[i,:,:],np.transpose(J[i,:,:]))))
        return target

    def plot(self):
        assert self.likelihood.Y.shape[1] == 2
        pb.scatter(self.likelihood.Y[:, 0], self.likelihood.Y[:, 1], 40, self.X[:, 0].copy(), linewidth=0, cmap=pb.cm.jet)  # @UndefinedVariable
        Xnew = np.linspace(self.X.min(), self.X.max(), 200)[:, None]
        mu, var, upper, lower = self.predict(Xnew)
        pb.plot(mu[:, 0], mu[:, 1], 'k', linewidth=1.5)

    def plot_latent(self, *args, **kwargs):
        return util.plot_latent.plot_latent(self, *args, **kwargs)

    def plot_magnification(self, *args, **kwargs):
        return util.plot_latent.plot_magnification(self, *args, **kwargs)
