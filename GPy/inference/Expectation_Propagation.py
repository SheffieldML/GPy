import numpy as np
import random
from scipy import stats, linalg
from .likelihoods import likelihood
from ..core import model
from ..util.linalg import pdinv,mdot,jitchol
from ..util.plot import gpplot
from .. import kern

class EP:
    def __init__(self,covariance,likelihood,Kmn=None,Knn_diag=None,epsilon=1e-3,powerep=[1.,1.]):
        """
        Expectation Propagation

        Arguments
        ---------
        X : input observations
        likelihood : Output's likelihood (likelihood class)
        kernel :  a GPy kernel (kern class)
        inducing : Either an array specifying the inducing points location or a sacalar defining their number. None value for using a non-sparse model is used.
        powerep : Power-EP parameters (eta,delta) - 2x1 numpy array (floats)
        epsilon : Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        """
        self.likelihood = likelihood
        assert covariance.shape[0] == covariance.shape[1]
        if Kmn is not None:
            self.Kmm = covariance
            self.Kmn = Kmn
            self.M = self.Kmn.shape[0]
            self.N = self.Kmn.shape[1]
            assert self.M < self.N, 'The number of inducing inputs must be smaller than the number of observations'
        else:
            self.K = covariance
            self.N = self.K.shape[0]
        if Knn_diag is not None:
            self.Knn_diag = Knn_diag
            assert len(Knn_diag) == self.N, 'Knn_diagonal has size different from N'

        self.epsilon = epsilon
        self.eta, self.delta = powerep
        self.jitter = 1e-12

        """
        Initial values - Likelihood approximation parameters:
        p(y|f) = t(f|tau_tilde,v_tilde)
        """
        self.tau_tilde = np.zeros(self.N)
        self.v_tilde = np.zeros(self.N)

    def restart_EP(self):
        """
        Set the EP approximation to initial state
        """
        self.tau_tilde = np.zeros(self.N)
        self.v_tilde = np.zeros(self.N)
        self.mu = np.zeros(self.N)

class Full(EP):
    def fit_EP(self):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006 (pag. 52-60)
        """
        #Prior distribution parameters: p(f|X) = N(f|0,K)
        #self.K = self.kernel.K(self.X,self.X)

        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        self.mu=np.zeros(self.N)
        self.Sigma=self.K.copy()

        """
        Initial values - Cavity distribution parameters:
        q_(f|mu_,sigma2_) = Product{q_i(f|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.N,dtype=float)
        self.v_ = np.empty(self.N,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.N,dtype=float)
        self.Z_hat = np.empty(self.N,dtype=float)
        phi = np.empty(self.N,dtype=float)
        mu_hat = np.empty(self.N,dtype=float)
        sigma2_hat = np.empty(self.N,dtype=float)

        #Approximation
        epsilon_np1 = self.epsilon + 1.
        epsilon_np2 = self.epsilon + 1.
       	self.iterations = 0
        self.np1 = [self.tau_tilde.copy()]
        self.np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.arange(self.N)
            random.shuffle(update_order)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./self.Sigma[i,i] - self.eta*self.tau_tilde[i]
                self.v_[i] = self.mu[i]/self.Sigma[i,i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.likelihood.moments_match(i,self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./self.Sigma[i,i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - self.mu[i]/self.Sigma[i,i])
                self.tau_tilde[i] = self.tau_tilde[i] + Delta_tau
                self.v_tilde[i] = self.v_tilde[i] + Delta_v
                #Posterior distribution parameters update
                si=self.Sigma[:,i].reshape(self.N,1)
                self.Sigma = self.Sigma - Delta_tau/(1.+ Delta_tau*self.Sigma[i,i])*np.dot(si,si.T)
                self.mu = np.dot(self.Sigma,self.v_tilde)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            Sroot_tilde_K = np.sqrt(self.tau_tilde)[:,None]*(self.K)
            B = np.eye(self.N) + np.sqrt(self.tau_tilde)[None,:]*Sroot_tilde_K
            L = jitchol(B)
            V,info = linalg.flapack.dtrtrs(L,Sroot_tilde_K,lower=1)
            self.Sigma = self.K - np.dot(V.T,V)
            self.mu = np.dot(self.Sigma,self.v_tilde)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.N
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

            self.np2.append(self.v_tilde.copy())

class FITC(EP):
    def fit_EP(self):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see Naish-Guzman and Holden, 2008.
        """

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = diag(Knn-Qnn) + Qnn, Qnn = Knm*Kmmi*Kmn
        """
        self.Kmmi, self.Kmm_hld = pdinv(self.Kmm)
        self.P0 = self.Kmn.T
        self.KmnKnm = np.dot(self.P0.T, self.P0)
        self.KmmiKmn = np.dot(self.Kmmi,self.P0.T)
        self.Qnn_diag = np.sum(self.P0.T*self.KmmiKmn,-2)
        self.Diag0 = self.Knn_diag - self.Qnn_diag
        self.R0 = jitchol(self.Kmmi).T

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*gamma
        """
        self.w = np.zeros(self.N)
        self.gamma = np.zeros(self.M)
        self.mu = np.zeros(self.N)
        self.P = self.P0.copy()
        self.R = self.R0.copy()
        self.Diag = self.Diag0.copy()
        self.Sigma_diag = self.Knn_diag

        """
        Initial values - Cavity distribution parameters:
        q_(g|mu_,sigma2_) = Product{q_i(g|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.N,dtype=float)
        self.v_ = np.empty(self.N,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.N,dtype=float)
        self.Z_hat = np.empty(self.N,dtype=float)
        phi = np.empty(self.N,dtype=float)
        mu_hat = np.empty(self.N,dtype=float)
        sigma2_hat = np.empty(self.N,dtype=float)

        #Approximation
        epsilon_np1 = 1
        epsilon_np2 = 1
       	self.iterations = 0
        self.np1 = [self.tau_tilde.copy()]
        self.np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.arange(self.N)
            random.shuffle(update_order)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./self.Sigma_diag[i] - self.eta*self.tau_tilde[i]
                self.v_[i] = self.mu[i]/self.Sigma_diag[i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.likelihood.moments_match(i,self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./self.Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - self.mu[i]/self.Sigma_diag[i])
                self.tau_tilde[i] = self.tau_tilde[i] + Delta_tau
                self.v_tilde[i] = self.v_tilde[i] + Delta_v
                #Posterior distribution parameters update
                dtd1 = Delta_tau*self.Diag[i] + 1.
                dii = self.Diag[i]
                self.Diag[i] = dii - (Delta_tau * dii**2.)/dtd1
                pi_ = self.P[i,:].reshape(1,self.M)
                self.P[i,:] = pi_ - (Delta_tau*dii)/dtd1 * pi_
                Rp_i = np.dot(self.R,pi_.T)
                RTR = np.dot(self.R.T,np.dot(np.eye(self.M) - Delta_tau/(1.+Delta_tau*self.Sigma_diag[i]) * np.dot(Rp_i,Rp_i.T),self.R))
                self.R = jitchol(RTR).T
                self.w[i] = self.w[i] + (Delta_v - Delta_tau*self.w[i])*dii/dtd1
                self.gamma = self.gamma + (Delta_v - Delta_tau*self.mu[i])*np.dot(RTR,self.P[i,:].T)
                self.RPT = np.dot(self.R,self.P.T)
                self.Sigma_diag = self.Diag + np.sum(self.RPT.T*self.RPT.T,-1)
                self.mu = self.w + np.dot(self.P,self.gamma)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            self.Diag = self.Diag0/(1.+ self.Diag0 * self.tau_tilde)
            self.P = (self.Diag / self.Diag0)[:,None] * self.P0
            self.RPT0 = np.dot(self.R0,self.P0.T)
            L = jitchol(np.eye(self.M) + np.dot(self.RPT0,(1./self.Diag0 - self.Diag/(self.Diag0**2))[:,None]*self.RPT0.T))
            self.R,info = linalg.flapack.dtrtrs(L,self.R0,lower=1)
            self.RPT = np.dot(self.R,self.P.T)
            self.Sigma_diag = self.Diag + np.sum(self.RPT.T*self.RPT.T,-1)
            self.w = self.Diag * self.v_tilde
            self.gamma = np.dot(self.R.T, np.dot(self.RPT,self.v_tilde))
            self.mu = self.w + np.dot(self.P,self.gamma)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.N
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.N
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())
