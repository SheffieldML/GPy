# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
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
    def __init__(self, Y, Q, init='PCA', X = None, kernel=None, **kwargs):
        if X is None:
            X = self.initialise_latent(init, Q, Y)
        if kernel is None:
            kernel = kern.rbf(Q) + kern.bias(Q)
        likelihood = Gaussian(Y)
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
        self.X = x[:self.X.size].reshape(self.N,self.Q).copy()
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

    def plot_latent(self,labels=None, which_indices=None, resolution=50):
        """
        :param labels: a np.array of size self.N containing labels for the points (can be number, strings, etc)
        :param resolution: the resolution of the grid on which to evaluate the predictive variance
        """

        util.plot.Tango.reset()
        
        if labels is None:
            labels = np.ones(self.N)
        if which_indices is None:
            if self.Q==1:
                input_1 = 0
                input_2 = None
            if self.Q==2:
                input_1, input_2 = 0,1
            else:
                #try to find a linear of RBF kern in the kernel
                k = [p for p in self.kern.parts if p.name in ['rbf','linear']]
                if (not len(k)==1) or (not k[0].ARD):
                    raise ValueError, "cannot Atomatically determine which dimensions to plot, please pass 'which_indices'"
                k = k[0]
                if k.name=='rbf':
                    input_1, input_2 = np.argsort(k.lengthscale)[:2]
                elif k.name=='linear':
                    input_1, input_2 = np.argsort(k.variances)[::-1][:2]

        #first, plot the output variance as a function of the latent space
        Xtest, xx,yy,xmin,xmax = util.plot.x_frame2D(self.X[:,[input_1, input_2]],resolution=resolution)
	Xtest_full = np.zeros((Xtest.shape[0], self.X.shape[1]))
	Xtest_full[:, :2] = Xtest
	mu, var, low, up = self.predict(Xtest_full)
	var = var[:, :2]
        pb.imshow(var.reshape(resolution,resolution).T[::-1,:],extent=[xmin[0],xmax[0],xmin[1],xmax[1]],cmap=pb.cm.binary,interpolation='bilinear')


        for i,ul in enumerate(np.unique(labels)):
            if type(ul) is np.string_:
                this_label = ul
            elif type(ul) is np.int64:
                this_label = 'class %i'%ul
            else:
                this_label = 'class %i'%i

            index = np.nonzero(labels==ul)[0]
            if self.Q==1:
                x = self.X[index,input_1]
                y = np.zeros(index.size)
            else:
                x = self.X[index,input_1]
                y = self.X[index,input_2]
            pb.plot(x,y,marker='o',color=util.plot.Tango.nextMedium(),mew=0,label=this_label,linewidth=0)

        pb.xlabel('latent dimension %i'%input_1)
        pb.ylabel('latent dimension %i'%input_2)

        if not np.all(labels==1.):
            pb.legend(loc=0,numpoints=1)

        pb.xlim(xmin[0],xmax[0])
        pb.ylim(xmin[1],xmax[1])

        return input_1, input_2
