# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from scipy import stats, linalg
from .. import kern
from ..inference.EP import Full
from ..inference.likelihoods import likelihood,probit,poisson,gaussian
from ..core import model
from ..util.linalg import pdinv,mdot #,jitchol
from ..util.plot import gpplot, Tango

class GP_EP2(model):
    def __init__(self,X,likelihood,kernel=None,normalize_X=False,Xslices=None,epsilon_ep=1e-3,epsion_em=.1,powerep=[1.,1.]):
        """
        Simple Gaussian Process with Non-Gaussian likelihood

        Arguments
        ---------
        :param X: input observations (NxD numpy.darray)
        :param likelihood: a GPy likelihood (likelihood class)
        :param kernel: a GPy kernel, defaults to rbf+white
        :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
        :type normalize_X: False|True
        :param epsilon_ep: convergence criterion for the Expectation Propagation algorithm, defaults to 1e-3
        :param powerep: power-EP parameters [$\eta$,$\delta$], defaults to [1.,1.] (list)
        :param Xslices: how the X,Y data co-vary in the kernel (i.e. which "outputs" they correspond to). See (link:slicing)
        :rtype: model object.
        """
        #.. Note:: Multiple independent outputs are allowed using columns of Y #TODO add this note?
        if kernel is None:
            kernel = kern.rbf(X.shape[1]) + kern.bias(X.shape[1]) + kern.white(X.shape[1])

        # parse arguments
        self.Xslices = Xslices
        assert isinstance(kernel, kern.kern)
        self.likelihood = likelihood
        #self.Y = self.likelihood.Y #we might not need this
        self.kern = kernel
        self.X = X
        assert len(self.X.shape)==2
        #assert len(self.Y.shape)==2
        #assert self.X.shape[0] == self.Y.shape[0]
        #self.N, self.D = self.Y.shape
        self.D = 1
        self.N, self.Q = self.X.shape

        #here's some simple normalisation
        if normalize_X:
            self._Xmean = X.mean(0)[None,:]
            self._Xstd = X.std(0)[None,:]
            self.X = (X.copy() - self._Xmean) / self._Xstd
            if hasattr(self,'Z'):
                self.Z = (self.Z - self._Xmean) / self._Xstd
        else:
            self._Xmean = np.zeros((1,self.X.shape[1]))
            self._Xstd = np.ones((1,self.X.shape[1]))

        #THIS PART IS NOT NEEDED
        """
        if normalize_Y:
            self._Ymean = Y.mean(0)[None,:]
            self._Ystd = Y.std(0)[None,:]
            self.Y = (Y.copy()- self._Ymean) / self._Ystd
        else:
            self._Ymean = np.zeros((1,self.Y.shape[1]))
            self._Ystd = np.ones((1,self.Y.shape[1]))

        if self.D > self.N:
            # then it's more efficient to store YYT
            self.YYT = np.dot(self.Y, self.Y.T)
        else:
            self.YYT = None
        """
        self.eta,self.delta = powerep
        self.epsilon_ep = epsilon_ep
        self.tau_tilde = np.zeros([self.N,self.D])
        self.v_tilde = np.zeros([self.N,self.D])
        model.__init__(self)

    def _set_params(self,p):
        self.kern._set_params_transformed(p)
        self.K = self.kern.K(self.X,slices1=self.Xslices)
        self.posterior_params()

    def _get_params(self):
        return self.kern._get_params_transformed()

    def _get_param_names(self):
        return self.kern._get_param_names_transformed()

    def approximate_likelihood(self):
        self.ep_approx = Full(self.K,self.likelihood,epsilon=self.epsilon_ep,powerep=[self.eta,self.delta])
        self.ep_approx.fit_EP()
        self.tau_tilde = self.ep_approx.tau_tilde[:,None]
        self.v_tilde = self.ep_approx.tau_tilde[:,None]
        self.posterior_params()
        self.Y = self.v_tilde/self.tau_tilde
        self._Ymean = np.zeros((1,self.Y.shape[1]))
        self._Ystd = np.ones((1,self.Y.shape[1]))
        #self.YYT = np.dot(self.Y, self.Y.T)

    def posterior_params(self):
        self.Sroot_tilde_K =  np.sqrt(self.tau_tilde.flatten())[:,None]*self.K
        B = np.eye(self.N) + np.sqrt(self.tau_tilde.flatten())[None,:]*self.Sroot_tilde_K
        self.Bi,self.L,self.Li,B_logdet = pdinv(B)
        V = np.dot(self.Li,self.Sroot_tilde_K)
        #V,info = linalg.flapack.dtrtrs(self.L,self.Sroot_tilde_K,lower=1)
        self.Sigma = self.K - np.dot(V.T,V)
        self.mu = np.dot(self.Sigma,self.v_tilde.flatten())


    #def _model_fit_term(self):
    #    """
    #    Computes the model fit using YYT if it's available
    #    """
    #    if self.YYT is None:
    #        return -0.5*np.sum(np.square(np.dot(self.Li,self.Y)))
    #    else:
    #        return -0.5*np.sum(np.multiply(self.Ki, self.YYT))

    def log_likelihood(self):
        mu_ = self.ep_approx.v_/self.ep_approx.tau_
        L1 =.5*sum(np.log(1+self.ep_approx.tau_tilde*1./self.ep_approx.tau_))-sum(np.log(np.diag(self.L)))
        L2A =.5*np.sum((self.Sigma-np.diag(1./(self.ep_approx.tau_+self.ep_approx.tau_tilde))) * np.dot(self.ep_approx.v_tilde[:,None],self.ep_approx.v_tilde[None,:]))
        L2B = .5*np.dot(mu_*(self.ep_approx.tau_/(self.ep_approx.tau_tilde+self.ep_approx.tau_)),self.ep_approx.tau_tilde*mu_ - 2*self.ep_approx.v_tilde)
        L3 = sum(np.log(self.ep_approx.Z_hat))
        return L1 + L2A + L2B + L3

    def dL_dK(self): #FIXME
        if self.YYT is None:
            alpha = np.dot(self.Ki,self.Y)
            dL_dK = 0.5*(np.dot(alpha,alpha.T)-self.D*self.Ki)
        else:
            dL_dK = 0.5*(mdot(self.Ki, self.YYT, self.Ki) - self.D*self.Ki)

        return dL_dK

    def _log_likelihood_gradients(self): #FIXME
        return self.kern.dK_dtheta(partial=self.dL_dK(),X=self.X)

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
        mu, var, phi = self._raw_predict(Xnew, slices, full_cov)

        #un-normalise
        mu = mu*self._Ystd + self._Ymean
        if full_cov:
            if self.D==1:
                var *= np.square(self._Ystd)
            else:
                var = var[:,:,None] * np.square(self._Ystd)
        else:
            if self.D==1:
                var *= np.square(np.squeeze(self._Ystd))
            else:
                var = var[:,None] * np.square(self._Ystd)

        return mu,var,phi

    def _raw_predict(self,_Xnew,slices, full_cov=False):
        """Internal helper function for making predictions, does not account for normalisation"""
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
        """
        K_x = self.kern.K(self.X,_Xnew)
        Kxx = self.kern.K(_Xnew)
        #aux1,info = linalg.flapack.dtrtrs(self.L,np.dot(self.Sroot_tilde_K,self.ep_approx.v_tilde),lower=1)
        #aux2,info = linalg.flapack.dtrtrs(self.L.T, aux1,lower=0)
        #aux2 = mdot(self.Li.T,self.Li,self.Sroot_tilde_K,self.ep_approx.v_tilde)
        aux2 = mdot(self.Bi,self.Sroot_tilde_K,self.ep_approx.v_tilde)
        zeta = np.sqrt(self.ep_approx.tau_tilde)*aux2
        f = np.dot(K_x.T,self.ep_approx.v_tilde-zeta)
       	#v,info = linalg.flapack.dtrtrs(self.L,np.sqrt(self.ep_approx.tau_tilde)[:,None]*K_x,lower=1)
        v = mdot(self.Li,np.sqrt(self.ep_approx.tau_tilde)[:,None]*K_x)
        variance = Kxx - np.dot(v.T,v)
       	vdiag = np.diag(variance)
        y=self.likelihood.predictive_mean(f,vdiag)
        return f,vdiag,y

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
            #m,v,phi = self.predict(Xnew,slices=which_functions)
            #gpplot(Xnew,m,v)
            mu_f, var_f, phi_f = self.predict(Xnew,slices=which_functions)
            pb.subplot(211)
            self.likelihood.plot1Da(X_new=Xnew,Mean_new=mu_f,Var_new=var_f,X_u=self.X,Mean_u=self.mu,Var_u=np.diag(self.Sigma))
            if samples:
                s = np.random.multivariate_normal(m.flatten(),v,samples)
                pb.plot(Xnew.flatten(),s.T, alpha = 0.4, c='#3465a4', linewidth = 0.8)
            pb.xlim(xmin,xmax)
            pb.subplot(212)
            self.likelihood.plot1Db(self.X,Xnew,phi_f)

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
