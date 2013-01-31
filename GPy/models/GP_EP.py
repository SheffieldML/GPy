# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from scipy import stats, linalg
from .. import kern
from ..inference.Expectation_Propagation import Full
from ..inference.likelihoods import likelihood,probit#,poisson,gaussian
from ..core import model
from ..util.linalg import pdinv,jitchol
from ..util.plot import gpplot

class GP_EP(model):
    def __init__(self,X,likelihood,kernel=None,epsilon_ep=1e-3,epsion_em=.1,powerep=[1.,1.]):
        """
        Simple Gaussian Process with Non-Gaussian likelihood

        Arguments
        ---------
        :param X: input observations (NxD numpy.darray)
        :param likelihood: a GPy likelihood (likelihood class)
        :param kernel: a GPy kernel (kern class)
        :param epsilon_ep: convergence criterion for the Expectation Propagation algorithm, defaults to 0.1 (float)
        :param powerep: power-EP parameters [$\eta$,$\delta$], defaults to [1.,1.] (list)
        :rtype: GPy model class.
        """
        if kernel is None:
            kernel = kern.rbf(X.shape[1]) + kern.bias(X.shape[1]) + kern.white(X.shape[1])

        assert isinstance(kernel,kern.kern), 'kernel is not a kern instance'
        self.likelihood = likelihood
        self.Y = self.likelihood.Y
        self.kernel = kernel
        self.X = X
        self.N, self.D = self.X.shape
        self.eta,self.delta = powerep
        self.epsilon_ep = epsilon_ep
        self.jitter = 1e-12
        self.K = self.kernel.K(self.X)
        model.__init__(self)

    def _set_params(self,p):
        self.kernel._set_params_transformed(p)

    def _get_params(self):
        return self.kernel._get_params_transformed()

    def _get_param_names(self):
        return self.kernel._get_param_names_transformed()

    def approximate_likelihood(self):
        self.ep_approx = Full(self.K,self.likelihood,epsilon=self.epsilon_ep,powerep=[self.eta,self.delta])
        self.ep_approx.fit_EP()

    def posterior_param(self):
        self.K = self.kernel.K(self.X)
        self.Sroot_tilde_K =  np.sqrt(self.ep_approx.tau_tilde)[:,None]*self.K
        B = np.eye(self.N) + np.sqrt(self.ep_approx.tau_tilde)[None,:]*self.Sroot_tilde_K
        #self.L = np.linalg.cholesky(B)
        self.L = jitchol(B)
        V,info = linalg.flapack.dtrtrs(self.L,self.Sroot_tilde_K,lower=1)
        self.Sigma = self.K - np.dot(V.T,V)
        self.mu = np.dot(self.Sigma,self.ep_approx.v_tilde)

    def log_likelihood(self):
        """
        Returns
        -------
        The EP approximation to the log-marginal likelihood
        """
        self.posterior_param()
        mu_ = self.ep_approx.v_/self.ep_approx.tau_
        L1 =.5*sum(np.log(1+self.ep_approx.tau_tilde*1./self.ep_approx.tau_))-sum(np.log(np.diag(self.L)))
        L2A =.5*np.sum((self.Sigma-np.diag(1./(self.ep_approx.tau_+self.ep_approx.tau_tilde))) * np.dot(self.ep_approx.v_tilde[:,None],self.ep_approx.v_tilde[None,:]))
        L2B = .5*np.dot(mu_*(self.ep_approx.tau_/(self.ep_approx.tau_tilde+self.ep_approx.tau_)),self.ep_approx.tau_tilde*mu_ - 2*self.ep_approx.v_tilde)
        L3 = sum(np.log(self.ep_approx.Z_hat))
        return L1 + L2A + L2B + L3

    def _log_likelihood_gradients(self):
        dK_dp = self.kernel.dK_dtheta(self.X)
        self.dK_dp = dK_dp
        aux1,info_1 = linalg.flapack.dtrtrs(self.L,np.dot(self.Sroot_tilde_K,self.ep_approx.v_tilde),lower=1)
        b = self.ep_approx.v_tilde - np.sqrt(self.ep_approx.tau_tilde)*linalg.flapack.dtrtrs(self.L.T,aux1)[0]
        U,info_u = linalg.flapack.dtrtrs(self.L,np.diag(np.sqrt(self.ep_approx.tau_tilde)),lower=1)
        dL_dK = 0.5*(np.outer(b,b)-np.dot(U.T,U))
        self.dL_dK = dL_dK
        return np.array([np.sum(dK_dpi*dL_dK) for dK_dpi in dK_dp.T])

    def predict(self,X):
        #TODO: check output dimensions
        self.posterior_param()
        K_x = self.kernel.K(self.X,X)
        Kxx = self.kernel.K(X)
        aux1,info = linalg.flapack.dtrtrs(self.L,np.dot(self.Sroot_tilde_K,self.ep_approx.v_tilde),lower=1)
        aux2,info = linalg.flapack.dtrtrs(self.L.T, aux1,lower=0)
        zeta = np.sqrt(self.ep_approx.tau_tilde)*aux2
        f = np.dot(K_x.T,self.ep_approx.v_tilde-zeta)
       	v,info = linalg.flapack.dtrtrs(self.L,np.sqrt(self.ep_approx.tau_tilde)[:,None]*K_x,lower=1)
        variance = Kxx - np.dot(v.T,v)
       	vdiag = np.diag(variance)
        y=self.likelihood.predictive_mean(f,vdiag)
        return f,vdiag,y

    def plot(self):
        """
        Plot the fitted model: training function values, inducing points used, mean estimate and confidence intervals.
        """
        if self.X.shape[1]==1:
            pb.figure()
            xmin,xmax = self.X.min(),self.X.max()
            xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
            Xnew = np.linspace(xmin,xmax,100)[:,None]
            mu_f, var_f, mu_phi = self.predict(Xnew)
            pb.subplot(211)
            self.likelihood.plot1Da(X_new=Xnew,Mean_new=mu_f,Var_new=var_f,X_u=self.X,Mean_u=self.mu,Var_u=np.diag(self.Sigma))
            pb.subplot(212)
            self.likelihood.plot1Db(self.X,Xnew,mu_phi)
        elif self.X.shape[1]==2:
            pb.figure()
            x1min,x1max = self.X[:,0].min(0),self.X[:,0].max(0)
            x2min,x2max = self.X[:,1].min(0),self.X[:,1].max(0)
            x1min, x1max = x1min-0.2*(x1max-x1min), x1max+0.2*(x1max-x1min)
            x2min, x2max = x2min-0.2*(x2max-x2min), x2max+0.2*(x1max-x1min)
            axis1 = np.linspace(x1min,x1max,50)
            axis2 = np.linspace(x2min,x2max,50)
            XX1, XX2 = [e.flatten() for e in np.meshgrid(axis1,axis2)]
            Xnew = np.c_[XX1.flatten(),XX2.flatten()]
            f,v,p = self.predict(Xnew)
            self.likelihood.plot2D(self.X,Xnew,p)
        else:
            raise NotImplementedError, "Cannot plot GPs with more than two input dimensions"

    def em(self,max_f_eval=1e4,epsilon=.1,plot_all=False): #TODO check this makes sense
        """
        Fits sparse_EP and optimizes the hyperparametes iteratively until convergence is achieved.
        """
        self.epsilon_em = epsilon
        log_likelihood_change = self.epsilon_em + 1.
        self.parameters_path = [self.kernel._get_params()]
        self.approximate_likelihood()
        self.site_approximations_path = [[self.ep_approx.tau_tilde,self.ep_approx.v_tilde]]
        self.log_likelihood_path = [self.log_likelihood()]
        iteration = 0
        while log_likelihood_change > self.epsilon_em:
            print 'EM iteration', iteration
            self.optimize(max_f_eval = max_f_eval)
            log_likelihood_new = self.log_likelihood()
            log_likelihood_change = log_likelihood_new - self.log_likelihood_path[-1]
            if log_likelihood_change < 0:
                print 'log_likelihood decrement'
                self.kernel._set_params_transformed(self.parameters_path[-1])
                self.kernM._set_params_transformed(self.parameters_path[-1])
            else:
                self.approximate_likelihood()
                self.log_likelihood_path.append(self.log_likelihood())
                self.parameters_path.append(self.kernel._get_params())
                self.site_approximations_path.append([self.ep_approx.tau_tilde,self.ep_approx.v_tilde])
            iteration += 1
