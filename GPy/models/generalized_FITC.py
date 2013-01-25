# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from scipy import stats, linalg
from .. import kern
from ..core import model
from ..util.linalg import pdinv,mdot
from ..util.plot import gpplot
#from ..inference.Expectation_Propagation import FITC
from ..inference.EP import FITC
from ..inference.likelihoods import likelihood,probit

class generalized_FITC(model):
    def __init__(self,X,likelihood,kernel=None,inducing=10,epsilon_ep=1e-3,powerep=[1.,1.]):
        """
        Naish-Guzman, A. and Holden, S. (2008) implemantation of EP with FITC.

        :param X: input observations
        :param likelihood: Output's likelihood (likelihood class)
        :param kernel: a GPy kernel
        :param inducing:  Either an array specifying the inducing points location or a scalar defining their number.
        :param epsilon_ep: EP convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :param powerep: Power-EP parameters (eta,delta) - 2x1 numpy array (floats)
        """
        assert isinstance(kernel,kern.kern)
        self.likelihood = likelihood
        self.Y = self.likelihood.Y
        self.kernel = kernel
        self.X = X
        self.N, self.D = self.X.shape
        assert self.Y.shape[0] == self.N
        if type(inducing) == int:
            self.M = inducing
            self.Z = (np.random.random_sample(self.D*self.M)*(self.X.max()-self.X.min())+self.X.min()).reshape(self.M,-1)
        elif type(inducing) == np.ndarray:
            self.Z = inducing
            self.M = self.Z.shape[0]
        self.eta,self.delta = powerep
        self.epsilon_ep = epsilon_ep
        self.jitter = 1e-12
        model.__init__(self)

    def _set_params(self,p):
        self.kernel._set_params_transformed(p[0:-self.Z.size])
        self.Z = p[-self.Z.size:].reshape(self.M,self.D)

    def _get_params(self):
        return np.hstack([self.kernel._get_params_transformed(),self.Z.flatten()])

    def _get_param_names(self):
        return self.kernel._get_param_names_transformed()+['iip_%i'%i for i in range(self.Z.size)]

    def approximate_likelihood(self):
        self.Kmm = self.kernel.K(self.Z)
        self.Knm = self.kernel.K(self.X,self.Z)
        self.Knn_diag = self.kernel.Kdiag(self.X)
        self.ep_approx = FITC(self.Kmm,self.likelihood,self.Knm.T,self.Knn_diag,epsilon=self.epsilon_ep,powerep=[self.eta,self.delta])
        self.ep_approx.fit_EP()

    def posterior_param(self):
        self.Knn_diag = self.kernel.Kdiag(self.X)
        self.Kmm = self.kernel.K(self.Z)
        self.Kmmi, self.Lmm, self.Lmmi, self.Kmm_logdet = pdinv(self.Kmm)
        self.Knm = self.kernel.K(self.X,self.Z)
        self.KmmiKmn = np.dot(self.Kmmi,self.Knm.T)
        self.Qnn = np.dot(self.Knm,self.KmmiKmn)
        self.Diag0 =  self.Knn_diag - np.diag(self.Qnn)
        self.R0 = np.linalg.cholesky(self.Kmmi).T

        self.Taut = self.ep_approx.tau_tilde/(1.+ self.ep_approx.tau_tilde*self.Diag0)
        self.KmnTaut = self.Knm.T*self.Taut[None,:]
        self.KmnTautKnm = np.dot(self.KmnTaut, self.Knm)
        self.Woodbury_inv, self.Wood_L, self.Wood_Li, self.Woodbury_logdet = pdinv(self.Kmm + self.KmnTautKnm)
        self.Qnn_diag = self.Knn_diag - np.diag(self.Qnn) + 1./self.ep_approx.tau_tilde
        self.Qi = -np.dot(self.KmnTaut.T, np.dot(self.Woodbury_inv,self.KmnTaut)) + np.diag(self.Taut)
        self.hld = 0.5*np.sum(np.log(self.Diag0 + 1./self.ep_approx.tau_tilde)) - 0.5*self.Kmm_logdet + 0.5*self.Woodbury_logdet

        self.Diag = self.Diag0/(1.+ self.Diag0 * self.ep_approx.tau_tilde)
        self.P = (self.Diag / self.Diag0)[:,None] * self.Knm
        self.RPT0 = np.dot(self.R0,self.Knm.T)
        self.L = np.linalg.cholesky(np.eye(self.M) + np.dot(self.RPT0,(1./self.Diag0 - self.Diag/(self.Diag0**2))[:,None]*self.RPT0.T))
        self.R,info = linalg.flapack.dtrtrs(self.L,self.R0,lower=1)
        self.RPT = np.dot(self.R,self.P.T)
        self.Sigma = np.diag(self.Diag) + np.dot(self.RPT.T,self.RPT)
        self.w = self.Diag * self.ep_approx.v_tilde
        self.gamma = np.dot(self.R.T, np.dot(self.RPT,self.ep_approx.v_tilde))
        self.mu = self.w + np.dot(self.P,self.gamma)
        self.mu_tilde = (self.ep_approx.v_tilde/self.ep_approx.tau_tilde)[:,None]

    def log_likelihood(self):
        self.posterior_param()
        self.YYT = np.dot(self.mu_tilde,self.mu_tilde.T)
        A = -self.hld
        B = -.5*np.sum(self.Qi*self.YYT)
        C = sum(np.log(self.ep_approx.Z_hat))
        D = .5*np.sum(np.log(1./self.ep_approx.tau_tilde + 1./self.ep_approx.tau_))
        E = .5*np.sum((self.ep_approx.v_/self.ep_approx.tau_ - self.mu_tilde.flatten())**2/(1./self.ep_approx.tau_ + 1./self.ep_approx.tau_tilde))
        return  A + B + C + D + E

    def _log_likelihood_gradients(self):
        dKmm_dtheta = self.kernel.dK_dtheta(self.Z)
        dKnn_dtheta = self.kernel.dK_dtheta(self.X)
        dKmn_dtheta = self.kernel.dK_dtheta(self.Z,self.X)
        dKmm_dZ = -self.kernel.dK_dX(self.Z)
        dKnm_dZ = -self.kernel.dK_dX(self.X,self.Z)
        tmp = [np.dot(dKmn_dtheta_i,self.KmmiKmn) for dKmn_dtheta_i in dKmn_dtheta.T]
        dQnn_dtheta = [tmp_i + tmp_i.T - np.dot(np.dot(self.KmmiKmn.T,dKmm_dtheta_i),self.KmmiKmn) for tmp_i,dKmm_dtheta_i in zip(tmp,dKmm_dtheta.T)]
        dDiag0_dtheta = [np.diag(dKnn_dtheta_i) - np.diag(dQnn_dtheta_i) for dKnn_dtheta_i,dQnn_dtheta_i in zip(dKnn_dtheta.T,dQnn_dtheta)]
        dQ_dtheta = [np.diag(dDiag0_dtheta_i) + dQnn_dtheta_i for dDiag0_dtheta_i,dQnn_dtheta_i in zip(dDiag0_dtheta,dQnn_dtheta)]
        dW_dtheta = [dKmm_dtheta_i + 2*np.dot(self.KmnTaut,dKmn_dtheta_i) - np.dot(self.KmnTaut*dDiag0_dtheta_i,self.KmnTaut.T) for dKmm_dtheta_i,dDiag0_dtheta_i,dKmn_dtheta_i in zip(dKmm_dtheta.T,dDiag0_dtheta,dKmn_dtheta.T)]

        QiY = np.dot(self.Qi, self.mu_tilde)
        QiYYQi = np.outer(QiY,QiY)
        WiKmnTaut = np.dot(self.Woodbury_inv,self.KmnTaut)
        K_Y = np.dot(self.KmmiKmn,QiY)
        # gradient - theta
        Atheta = [-0.5*np.dot(self.Taut,dDiag0_dtheta_i) + 0.5*np.sum(self.Kmmi*dKmm_dtheta_i) - 0.5*np.sum(self.Woodbury_inv*dW_dtheta_i) for dDiag0_dtheta_i,dKmm_dtheta_i,dW_dtheta_i in zip(dDiag0_dtheta,dKmm_dtheta.T,dW_dtheta)]
        Btheta = np.array([0.5*np.sum(QiYYQi*dQ_dtheta_i) for dQ_dtheta_i in dQ_dtheta])
        dL_dtheta = Atheta + Btheta
        # gradient - Z
        # Az
        dQnn_dZ_diag_a2 = (np.array([d[:,:,None]*self.KmmiKmn[:,:,None] for d in dKnm_dZ.transpose(2,0,1)]).reshape(self.D,self.M,self.N)).transpose(1,2,0)
        dQnn_dZ_diag_b2 = (np.array([(self.KmmiKmn*np.sum(d[:,:,None]*self.KmmiKmn,-2))[:,:,None] for d in dKmm_dZ.transpose(2,0,1)]).reshape(self.D,self.M,self.N)).transpose(1,2,0)
        dQnn_dZ_diag = dQnn_dZ_diag_a2 - dQnn_dZ_diag_b2
        d_hld_Diag1_dZ = -np.sum(np.dot(self.KmmiKmn*self.Taut,self.KmmiKmn.T)[:,:,None]*dKmm_dZ,-2) + np.sum((self.KmmiKmn*self.Taut)[:,:,None]*dKnm_dZ,-2)
        d_hld_Kmm_dZ = np.sum(self.Kmmi[:,:,None]*dKmm_dZ,-2)
        d_hld_W_dZ1 = np.sum(WiKmnTaut[:,:,None]*dKnm_dZ,-2)
        d_hld_W_dZ3 = np.sum(self.Woodbury_inv[:,:,None]*dKmm_dZ,-2)
        d_hld_W_dZ2 = np.array([np.sum(np.sum(WiKmnTaut.T*d[:,:,None]*self.KmnTaut.T,-2),-1) for d in dQnn_dZ_diag.transpose(2,0,1)]).T
        Az = d_hld_Diag1_dZ + d_hld_Kmm_dZ - d_hld_W_dZ1 - d_hld_W_dZ2 - d_hld_W_dZ3
        # Bz
        Bz2 = np.sum(np.dot(K_Y,QiY.T)[:,:,None]*dKnm_dZ,-2)
        Bz3 = - np.sum(np.dot(K_Y,K_Y.T)[:,:,None]*dKmm_dZ,-2)
        Bz1 = -np.array([np.sum((QiY**2)*d[:,:,None],-2) for d in dQnn_dZ_diag.transpose(2,0,1)]).reshape(self.D,self.M).T
        Bz = Bz1 + Bz2 + Bz3
        dL_dZ = (Az + Bz).flatten()
        return np.hstack([dL_dtheta, dL_dZ])

    def predict(self,X):
        """
        Make a prediction for the vsGP model

        Arguments
        ---------
        X : Input prediction data - Nx1 numpy array (floats)
        """
        #TODO: check output dimensions
        K_x = self.kernel.K(self.Z,X)
        Kxx = self.kernel.K(X)
        #K_x = self.kernM.cross.K(X)
        # q(u|f) = N(u| R0i*mu_u*f, R0i*C*R0i.T)

        # Ci = I + (RPT0)Di(RPT0).T
        # C = I - [RPT0] * (D+[RPT0].T*[RPT0])^-1*[RPT0].T
        #   = I - [RPT0] * (D + self.Qnn)^-1 * [RPT0].T
        #   = I - [RPT0] * (U*U.T)^-1 * [RPT0].T
        #   = I - V.T * V
        U = np.linalg.cholesky(np.diag(self.Diag0) + self.Qnn)
        V,info = linalg.flapack.dtrtrs(U,self.RPT0.T,lower=1)
        C = np.eye(self.M) - np.dot(V.T,V)
        mu_u = np.dot(C,self.RPT0)*(1./self.Diag0[None,:])
        #self.C = C
        #self.RPT0 = np.dot(self.R0,self.Knm.T) P0.T
        #self.mu_u = mu_u
        #self.U = U
        # q(u|y) = N(u| R0i*mu_H,R0i*Sigma_H*R0i.T)
        mu_H = np.dot(mu_u,self.mu)
        self.mu_H = mu_H
        Sigma_H = C + np.dot(mu_u,np.dot(self.Sigma,mu_u.T))
        # q(f_star|y) = N(f_star|mu_star,sigma2_star)
        KR0T = np.dot(K_x.T,self.R0.T)
        mu_star = np.dot(KR0T,mu_H)
        sigma2_star = Kxx + np.dot(KR0T,np.dot(Sigma_H - np.eye(self.M),KR0T.T))
        vdiag = np.diag(sigma2_star)
        # q(y_star|y) = non-gaussian posterior probability of class membership
        p = self.likelihood.predictive_mean(mu_star,vdiag)
        return mu_star,vdiag,p

    def plot(self):
        """
        Plot the fitted model: training function values, inducing points used, mean estimate and confidence intervals.
        """
        if self.X.shape[1]==1:
            pb.figure()
            xmin,xmax = np.r_[self.X,self.Z].min(),np.r_[self.X,self.Z].max()
            xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
            Xnew = np.linspace(xmin,xmax,100)[:,None]
            mu_f, var_f, mu_phi = self.predict(Xnew)
            self.mu_inducing,self.var_diag_inducing,self.phi_inducing = self.predict(self.Z)
            pb.subplot(211)
            self.likelihood.plot1Da(X_new=Xnew,Mean_new=mu_f,Var_new=var_f,X_u=self.Z,Mean_u=self.mu_inducing,Var_u=self.var_diag_inducing)
            pb.subplot(212)
            self.likelihood.plot1Db(self.X,Xnew,mu_phi,self.Z)
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
            self.likelihood.plot2D(self.X,Xnew,p,self.Z)
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
        self.inducing_inputs_path = [self.Z]
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
                self.kernM = self.kernel.copy()
                slef.kernM.expand_X(self.iducing_inputs_path[-1])
                self.__init__(self.kernel,self.likelihood,kernM=self.kernM,powerep=[self.eta,self.delta],epsilon_ep = self.epsilon_ep, epsilon_em = self.epsilon_em)

            else:
                self.approximate_likelihood()
                self.log_likelihood_path.append(self.log_likelihood())
                self.parameters_path.append(self.kernel._get_params())
                self.site_approximations_path.append([self.ep_approx.tau_tilde,self.ep_approx.v_tilde])
                self.inducing_inputs_path.append(self.Z)
            iteration += 1
