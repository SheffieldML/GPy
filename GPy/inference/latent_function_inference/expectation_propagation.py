# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from ...util.linalg import jitchol, DSYR, dtrtrs, dtrtri
from ...core.parameterization.observable_array import ObsAr
from . import ExactGaussianInference, VarDTC
from ...util import diag

log_2_pi = np.log(2*np.pi)

class EPBase(object):
    def __init__(self, epsilon=1e-6, eta=1., delta=1.):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006.

        :param epsilon: Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :type epsilon: float
        :param eta: parameter for fractional EP updates.
        :type eta: float64
        :param delta: damping EP updates factor.
        :type delta: float64
        """
        super(EPBase, self).__init__()
        self.epsilon, self.eta, self.delta = epsilon, eta, delta
        self.reset()

    def reset(self):
        self.old_mutilde, self.old_vtilde = None, None
        self._ep_approximation = None

    def on_optimization_start(self):
        self._ep_approximation = None

    def on_optimization_end(self):
        # TODO: update approximation in the end as well? Maybe even with a switch?
        pass

class EP(EPBase, ExactGaussianInference):
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, precision=None, K=None):
        num_data, output_dim = Y.shape
        assert output_dim ==1, "ep in 1D only (for now!)"

        if K is None:
            K = kern.K(X)

        if self._ep_approximation is None:
            #if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation = self.expectation_propagation(K, Y, likelihood, Y_metadata)
        else:
            #if we've already run EP, just use the existing approximation stored in self._ep_approximation
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation

        return super(EP, self).inference(kern, X, likelihood, mu_tilde[:,None], mean_function=mean_function, Y_metadata=Y_metadata, precision=1./tau_tilde, K=K)

    def expectation_propagation(self, K, Y, likelihood, Y_metadata):

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"


        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(num_data)
        Sigma = K.copy()
        diag.add(Sigma, 1e-7)

        #Initial values - Marginal moments
        Z_hat = np.empty(num_data,dtype=np.float64)
        mu_hat = np.empty(num_data,dtype=np.float64)
        sigma2_hat = np.empty(num_data,dtype=np.float64)

        #initial values - Gaussian factors
        if self.old_mutilde is None:
            tau_tilde, mu_tilde, v_tilde = np.zeros((3, num_data))
        else:
            assert self.old_mutilde.size == num_data, "data size mis-match: did you change the data? try resetting!"
            mu_tilde, v_tilde = self.old_mutilde, self.old_vtilde
            tau_tilde = v_tilde/mu_tilde

        #Approximation
        tau_diff = self.epsilon + 1.
        v_diff = self.epsilon + 1.
        iterations = 0
        while (tau_diff > self.epsilon) or (v_diff > self.epsilon):
            update_order = np.random.permutation(num_data)
            for i in update_order:
                #Cavity distribution parameters
                tau_cav = 1./Sigma[i,i] - self.eta*tau_tilde[i]
                v_cav = mu[i]/Sigma[i,i] - self.eta*v_tilde[i]
                #Marginal moments
                Z_hat[i], mu_hat[i], sigma2_hat[i] = likelihood.moments_match_ep(Y[i], tau_cav, v_cav)#, Y_metadata=None)#=(None if Y_metadata is None else Y_metadata[i]))
                #Site parameters update
                delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma[i,i])
                delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma[i,i])
                tau_tilde[i] += delta_tau
                v_tilde[i] += delta_v
                #Posterior distribution parameters update
                DSYR(Sigma, Sigma[:,i].copy(), -delta_tau/(1.+ delta_tau*Sigma[i,i]))
                mu = np.dot(Sigma, v_tilde)

            #(re) compute Sigma and mu using full Cholesky decompy
            tau_tilde_root = np.sqrt(tau_tilde)
            Sroot_tilde_K = tau_tilde_root[:,None] * K
            B = np.eye(num_data) + Sroot_tilde_K * tau_tilde_root[None,:]
            L = jitchol(B)
            V, _ = dtrtrs(L, Sroot_tilde_K, lower=1)
            Sigma = K - np.dot(V.T,V)
            mu = np.dot(Sigma,v_tilde)

            #monitor convergence
            if iterations>0:
                tau_diff = np.mean(np.square(tau_tilde-tau_tilde_old))
                v_diff = np.mean(np.square(v_tilde-v_tilde_old))
            tau_tilde_old = tau_tilde.copy()
            v_tilde_old = v_tilde.copy()

            iterations += 1

        mu_tilde = v_tilde/tau_tilde
        return mu, Sigma, mu_tilde, tau_tilde, Z_hat

class EPDTC(EPBase, VarDTC):
    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, Lm=None, dL_dKmm=None, psi0=None, psi1=None, psi2=None):
        assert Y.shape[1]==1, "ep in 1D only (for now!)"

        Kmm = kern.K(Z)
        if psi1 is None:
            try:
                Kmn = kern.K(Z, X)
            except TypeError:
                Kmn = kern.psi1(Z, X).T
        else:
            Kmn = psi1.T

        if self._ep_approximation is None:
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
        else:
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation

        return super(EPDTC, self).inference(kern, X, Z, likelihood, mu_tilde,
                                            mean_function=mean_function,
                                            Y_metadata=Y_metadata,
                                            precision=tau_tilde,
                                            Lm=Lm, dL_dKmm=dL_dKmm,
                                            psi0=psi0, psi1=psi1, psi2=psi2)

    def expectation_propagation(self, Kmm, Kmn, Y, likelihood, Y_metadata):
        num_data, output_dim = Y.shape
        assert output_dim == 1, "This EP methods only works for 1D outputs"

        LLT0 = Kmm.copy()
        #diag.add(LLT0, 1e-8)

        Lm = jitchol(LLT0)
        Lmi = dtrtri(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        KmmiKmn = np.dot(Kmmi,Kmn)
        Qnn_diag = np.sum(Kmn*KmmiKmn,-2)

        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(num_data)
        LLT = Kmm.copy() #Sigma = K.copy()
        Sigma_diag = Qnn_diag.copy() + 1e-8

        #Initial values - Marginal moments
        Z_hat = np.zeros(num_data,dtype=np.float64)
        mu_hat = np.zeros(num_data,dtype=np.float64)
        sigma2_hat = np.zeros(num_data,dtype=np.float64)

        #initial values - Gaussian factors
        if self.old_mutilde is None:
            tau_tilde, mu_tilde, v_tilde = np.zeros((3, num_data))
        else:
            assert self.old_mutilde.size == num_data, "data size mis-match: did you change the data? try resetting!"
            mu_tilde, v_tilde = self.old_mutilde, self.old_vtilde
            tau_tilde = v_tilde/mu_tilde

        #Approximation
        tau_diff = self.epsilon + 1.
        v_diff = self.epsilon + 1.
        iterations = 0
        tau_tilde_old = 0.
        v_tilde_old = 0.
        update_order = np.random.permutation(num_data)

        while (tau_diff > self.epsilon) or (v_diff > self.epsilon):
            for i in update_order:
                #Cavity distribution parameters
                tau_cav = 1./Sigma_diag[i] - self.eta*tau_tilde[i]
                v_cav = mu[i]/Sigma_diag[i] - self.eta*v_tilde[i]
                #Marginal moments
                Z_hat[i], mu_hat[i], sigma2_hat[i] = likelihood.moments_match_ep(Y[i], tau_cav, v_cav)#, Y_metadata=None)#=(None if Y_metadata is None else Y_metadata[i]))
                #Site parameters update
                delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                tau_tilde[i] += delta_tau
                v_tilde[i] += delta_v
                #Posterior distribution parameters update

                #DSYR(Sigma, Sigma[:,i].copy(), -delta_tau/(1.+ delta_tau*Sigma[i,i]))
                DSYR(LLT,Kmn[:,i].copy(),delta_tau)
                L = jitchol(LLT+np.eye(LLT.shape[0])*1e-7)

                V,info = dtrtrs(L,Kmn,lower=1)
                Sigma_diag = np.sum(V*V,-2)
                si = np.sum(V.T*V[:,i],-1)
                mu += (delta_v-delta_tau*mu[i])*si
                #mu = np.dot(Sigma, v_tilde)

            #(re) compute Sigma and mu using full Cholesky decompy
            LLT = LLT0 + np.dot(Kmn*tau_tilde[None,:],Kmn.T)
            #diag.add(LLT, 1e-8)
            L = jitchol(LLT)
            V, _ = dtrtrs(L,Kmn,lower=1)
            V2, _ = dtrtrs(L.T,V,lower=0)
            #Sigma_diag = np.sum(V*V,-2)
            #Knmv_tilde = np.dot(Kmn,v_tilde)
            #mu = np.dot(V2.T,Knmv_tilde)
            Sigma = np.dot(V2.T,V2)
            mu = np.dot(Sigma,v_tilde)

            #monitor convergence
            #if iterations>0:
            tau_diff = np.mean(np.square(tau_tilde-tau_tilde_old))
            v_diff = np.mean(np.square(v_tilde-v_tilde_old))

            tau_tilde_old = tau_tilde.copy()
            v_tilde_old = v_tilde.copy()

            # Only to while loop once:?
            tau_diff = 0
            v_diff = 0
            iterations += 1

        mu_tilde = v_tilde/tau_tilde
        return mu, Sigma, ObsAr(mu_tilde[:,None]), tau_tilde, Z_hat

