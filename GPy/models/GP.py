# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from .. import kern
from ..core import model
from ..util.linalg import pdinv,mdot
from ..util.plot import gpplot, Tango
from ..inference.EP import Full # TODO: tidy
from ..inference import likelihoods

class GP(model):
    """
    Gaussian Process model for regression and EP

    :param X: input observations
    :param kernel: a GPy kernel, defaults to rbf+white
    :parm likelihood: a GPy likelihood
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :param Xslices: how the X,Y data co-vary in the kernel (i.e. which "outputs" they correspond to). See (link:slicing)
    :rtype: model object
    :param epsilon_ep: convergence criterion for the Expectation Propagation algorithm, defaults to 0.1
    :param powerep: power-EP parameters [$\eta$,$\delta$], defaults to [1.,1.]
    :type powerep: list

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """
    #TODO: when using EP, predict needs to return 3 values otherwise it just needs 2. At the moment predict returns 3 values in any case.

    def __init__(self, X, kernel, likelihood, normalize_X=False, Xslices=None):

        # parse arguments
        self.Xslices = Xslices
        self.X = X
        assert len(self.X.shape)==2
        self.N, self.Q = self.X.shape
        assert isinstance(kernel, kern.kern)
        self.kern = kernel

        #here's some simple normalisation for the inputs
        if normalize_X:
            self._Xmean = X.mean(0)[None,:]
            self._Xstd = X.std(0)[None,:]
            self.X = (X.copy() - self._Xmean) / self._Xstd
            if hasattr(self,'Z'):
                self.Z = (self.Z - self._Xmean) / self._Xstd
        else:
            self._Xmean = np.zeros((1,self.X.shape[1]))
            self._Xstd = np.ones((1,self.X.shape[1]))

        self.likelihood = likelihood
        self.Y = self.likelihood.Y
        self.YYT = self.likelihood.YYT # TODO: this is ugly. what about sufficient_stats?
        assert self.X.shape[0] == self.likelihood.Y.shape[0]
        self.N, self.D = self.likelihood.Y.shape

        model.__init__(self)

    def _set_params(self,p):
        self.kern._set_params_transformed(p[:self.kern.Nparam])
        self.likelihood._set_params(p[self.kern.Nparam:])

        self.K = self.kern.K(self.X,slices1=self.Xslices)
        self.K += np.diag(self.likelihood_variance)

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        #the gradient of the likelihood wrt the covariance matrix
        if self.YYT is None:
            self._alpha = np.dot(self.Ki,self.Y)
            self._alpha2 = np.square(self._alpha)
            self.dL_dK = 0.5*(np.dot(self._alpha,self._alpha.T)-self.D*self.Ki)
        else:
            tmp = mdot(self.Ki, self.YYT, self.Ki)
            self._alpha2 = np.diag(tmp)
            self.dL_dK = 0.5*(tmp - self.D*self.Ki)

    def _get_params(self):
        return np.hstack((self.kern._get_params_transformed(), self.likelihood._get_params()))

    def _get_param_names(self):
        return self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian (or direct: TODO) likelihood, no iteration is required:
        this function does nothing
        """
        self.likelihood.fit(self.K)
        self.Y, self.YYT, self.likelihood_variance, self.likelihood_Z = self.likelihood.sufficient_stats() # TODO: just store these in the likelihood?

    def _model_fit_term(self):
        """
        Computes the model fit using YYT if it's available
        """
        if self.YYT is None:
            return -0.5*np.sum(np.square(np.dot(self.Li,self.Y)))
        else:
            return -0.5*np.sum(np.multiply(self.Ki, self.YYT))

    def log_likelihood(self):
        """
        The log marginal likelihood of the GP.

        For an EP model,  can be written as the log likelihood of a regression
        model for a new variable Y* = v_tilde/tau_tilde, with a covariance
        matrix K* = K + diag(1./tau_tilde) plus a normalization term.
        """
        return -0.5*self.D*self.K_logdet + self.model_fit_term() + self.likelihood.Z


    def _log_likelihood_gradients(self):
        """
        The gradient of all parameters.

        For the kernel parameters, use the chain rule via dL_dK

        For the likelihood parameters, pass in alpha = K^-1 y
        """
        return np.hstack((self.kern.dK_dtheta(partial=self.dL_dK(),X=self.X), self.likelihood._gradients(self.alpha2)))

    def _raw_predict(self,_Xnew,slices, full_cov=False):
        """
        Internal helper function for making predictions, does not account
        for normalisation or likelihood
        """
        Kx = self.kern.K(self.X,_Xnew, slices1=self.Xslices,slices2=slices)
        mu = np.dot(np.dot(Kx.T,self.Ki),self.Y)
        KiKx = np.dot(self.Ki,Kx)
        if full_cov:
            Kxx = self.kern.K(_Xnew, slices1=slices,slices2=slices)
            var = Kxx - np.dot(KiKx.T,Kx)
        else:
            Kxx = self.kern.Kdiag(_Xnew, slices=slices)
            var = Kxx - np.sum(np.multiply(KiKx,Kx),0)
        return mu, var


    def predict(self,Xnew, slices=None, full_cov=False):
        """
        Predict the function(s) at the new point(s) Xnew.

        Arguments
        ---------
        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.Q
        :param slices:  specifies which outputs kernel(s) the Xnew correspond to (see below)
        :type slices: (None, list of slice objects, list of ints)
        :param full_cov: whether to return the folll covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.D
        :rtype: posterior variance, a Numpy array, Nnew x Nnew x (self.D)

        .. Note:: "slices" specifies how the the points X_new co-vary wich the training points.

             - If None, the new points covary throigh every kernel part (default)
             - If a list of slices, the i^th slice specifies which data are affected by the i^th kernel part
             - If a list of booleans, specifying which kernel parts are active

           If full_cov and self.D > 1, the return shape of var is Nnew x Nnew x self.D. If self.D == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalisations of the output dimensions.

        """
        #normalise X values
        Xnew = (Xnew.copy() - self._Xmean) / self._Xstd
        mu, var, phi = self._raw_predict(Xnew, slices, full_cov=full_cov)

        #now push through likelihood TODO

        return mean, _5pc, _95pc

    def plot(self,samples=0,plot_limits=None,which_data='all',which_functions='all',resolution=None,full_cov=False):
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
        Yorig = Y*self._Ystd + self._Ymean #NOTE For EP this is v_tilde/beta

        if plot_limits is None:
            xmin,xmax = Xorig.min(0),Xorig.max(0)
            xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
        elif len(plot_limits)==2:
            xmin, xmax = plot_limits
        else:
            raise ValueError, "Bad limits for plotting"

        if self.X.shape[1]==1:
            Xnew = np.linspace(xmin,xmax,resolution or 200)[:,None]
            m,v = self.predict(Xnew,slices=which_functions,full_cov=full_cov)
            if self.EP:
                pb.subplot(211)
            gpplot(Xnew,m,v)
            if samples: #NOTE why don't we put samples as a parameter of gpplot
                s = np.random.multivariate_normal(m.flatten(),np.diag(v.flatten()),samples)
                pb.plot(Xnew.flatten(),s.T, alpha = 0.4, c='#3465a4', linewidth = 0.8)
            pb.plot(Xorig,Yorig,'kx',mew=1.5)
            pb.xlim(xmin,xmax)

            if self.EP:
                phi_m, phi_v, phi_l, phi_u = self.likelihood.predictive_values(m,v)

        elif self.X.shape[1]==2:
            resolution = 50 or resolution
            xx,yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
            Xtest = np.vstack((xx.flatten(),yy.flatten())).T
            zz,vv,phi = self.predict(Xtest,slices=which_functions,full_cov=full_cov)
            zz = zz.reshape(resolution,resolution)
            pb.contour(xx,yy,zz,vmin=zz.min(),vmax=zz.max(),cmap=pb.cm.jet)
            pb.scatter(Xorig[:,0],Xorig[:,1],40,Yorig,linewidth=0,cmap=pb.cm.jet,vmin=zz.min(),vmax=zz.max())
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])

        else:
            raise NotImplementedError, "Cannot plot GPs with more than two input dimensions"
