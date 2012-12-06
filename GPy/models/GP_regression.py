# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from .. import kern
from ..core import model
from ..util.linalg import pdinv,mdot
from ..util.plot import gpplot, Tango

class GP_regression(model):
    """
    Gaussian Process model for regression

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

    def __init__(self,X,Y,kernel=None,normalize_X=False,normalize_Y=False, Xslices=None):
        if kernel is None:
            kernel = kern.rbf(X.shape[1]) + kern.bias(X.shape[1]) + kern.white(X.shape[1])

        # parse arguments
        self.Xslices = Xslices
        assert isinstance(kernel, kern.kern)
        self.kern = kernel
        self.X = X
        self.Y = Y
        assert len(self.X.shape)==2
        assert len(self.Y.shape)==2
        assert self.X.shape[0] == self.Y.shape[0]
        self.N, self.D = self.Y.shape
        self.N, self.Q = self.X.shape

        #here's some simple normalisation
        if normalize_X:
            self._Xmean = X.mean(0)[None,:]
            self._Xstd = X.std(0)[None,:]
            self.X = (X.copy()- self._Xmean) / self._Xstd
        else:
            self._Xmean = np.zeros((1,self.X.shape[1]))
            self._Xstd = np.ones((1,self.X.shape[1]))

        if normalize_Y:
            self._Ymean = Y.mean(0)[None,:]
            self._Ystd = Y.std(0)[None,:]
            self.Y = (Y.copy()- self._Ymean) / self._Ystd
        else:
            self._Ymean = np.zeros((1,self.Y.shape[1]))
            self._Ystd = np.ones((1,self.Y.shape[1]))

        if self.D > self.N:
            # then it's more efficient to store Youter
            self.Youter = np.dot(self.Y, self.Y.T)
        else:
            self.Youter = None

        model.__init__(self)

    def set_param(self,p):
        self.kern.expand_param(p)
        self.K = self.kern.K(self.X,slices1=self.Xslices)
        self.Ki,self.hld = pdinv(self.K)

    def get_param(self):
        return self.kern.extract_param()

    def get_param_names(self):
        return self.kern.extract_param_names()

    def _model_fit_term(self):
        """
        Computes the model fit using Youter if it's available
        """

        if self.Youter is None:
            return -0.5*np.trace(mdot(self.Y.T,self.Ki,self.Y))
        else:
            return -0.5*np.sum(np.multiply(self.Ki, self.Youter))

    def log_likelihood(self):
        complexity_term = -0.5*self.N*self.D*np.log(2.*np.pi) - self.D*self.hld
        return complexity_term + self._model_fit_term()

    def dL_dK(self):
        if self.Youter is None:
            alpha = np.dot(self.Ki,self.Y)
            dL_dK = 0.5*(np.dot(alpha,alpha.T)-self.D*self.Ki)
        else:
            dL_dK = 0.5*(mdot(self.Ki, self.Youter, self.Ki) - self.D*self.Ki)

        return dL_dK

    def log_likelihood_gradients(self):
        return self.kern.dK_dtheta(partial=self.dL_dK(),X=self.X)

    def predict(self,Xnew, slices=None):
        """

        Predict the function(s) at the new point(s) Xnew.

        Arguments
        ---------
        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.Q
        :param slices:  specifies which outputs kernel(s) the Xnew correspond to (see below)
        :type slices: (None, list of slice objects, list of ints)
        :rtype: posterior mean,  a Numpy array, Nnew x self.D
        :rtype: posterior variance, a Numpy array, Nnew x Nnew x (self.D)

        .. Note:: "slices" specifies how the the points X_new co-vary wich the training points.

             - If None, the new points covary throigh every kernel part (default)
             - If a list of slices, the i^th slice specifies which data are affected by the i^th kernel part
             - If a list of booleans, specifying which kernel parts are active

           If self.D > 1, the return shape of var is Nnew x Nnew x self.D. If self.D == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalisations of the output dimensions.

        """

        #normalise X values
        Xnew = (Xnew.copy() - self._Xmean) / self._Xstd
        mu, var = self._raw_predict(Xnew,slices)

        #un-normalise
        mu = mu*self._Ystd + self._Ymean
        if self.D==1:
            var *= np.square(self._Ystd)
        else:
            var = var[:,:,None] * np.square(self._Ystd)
        return mu,var

    def _raw_predict(self,_Xnew,slices):
        """Internal helper function for making predictions, does not account for normalisation"""
        Kx = self.kern.K(self.X,_Xnew, slices1=self.Xslices,slices2=slices)
        Kxx = self.kern.K(_Xnew, slices1=slices,slices2=slices)
        mu = np.dot(np.dot(Kx.T,self.Ki),self.Y)
        var = Kxx - np.dot(np.dot(Kx.T,self.Ki),Kx)
        return mu, var

    def plot(self,samples=0,plot_limits=None,which_data='all',which_functions='all',resolution=None):
        """
        :param samples: the number of a posteriori samples to plot
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_functions: which of the kernel functions to plot (additively)
        :type which_functions: list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D

        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, we've no implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions using which_data and which_functions
        """
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)

        X = self.X[which_data,:]
        Y = self.Y[which_data,:]

        Xorig = X*self._Xstd + self._Xmean
        Yorig = Y*self._Ystd + self._Ymean
        if plot_limits is None:
            xmin,xmax = Xorig.min(0),Xorig.max(0)
            xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
        elif len(plot_limits)==2:
            xmin, xmax = plot_limits
        else:
            raise ValueError, "Bad limits for plotting"


        if self.X.shape[1]==1:
            Xnew = np.linspace(xmin,xmax,resolution or 200)[:,None]
            m,v = self.predict(Xnew,slices=which_functions)
            gpplot(Xnew,m,v)
            if samples:
                s = np.random.multivariate_normal(m.flatten(),v,samples)
                pb.plot(Xnew.flatten(),s.T, alpha = 0.4, c='#3465a4', linewidth = 0.8)
            pb.plot(Xorig,Yorig,'kx',mew=1.5)
            pb.xlim(xmin,xmax)

        elif self.X.shape[1]==2:
            resolution = 50 or resolution
            xx,yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
            Xtest = np.vstack((xx.flatten(),yy.flatten())).T
            zz,vv = self.predict(Xtest,slices=which_functions)
            zz = zz.reshape(resolution,resolution)
            pb.contour(xx,yy,zz,vmin=zz.min(),vmax=zz.max(),cmap=pb.cm.jet)
            pb.scatter(Xorig[:,0],Xorig[:,1],40,Yorig,linewidth=0,cmap=pb.cm.jet,vmin=zz.min(),vmax=zz.max())
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])

        else:
            raise NotImplementedError, "Cannot plot GPs with more than two input dimensions"
