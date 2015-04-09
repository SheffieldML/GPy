# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from ...util.linalg import pdinv,jitchol,DSYR,tdot,dtrtrs, dpotrs
from .posterior import Posterior
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class EP(LatentFunctionInference):
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

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, Z=None):
        assert mean_function is None, "inference with a mean function not implemented"
        num_data, output_dim = Y.shape
        assert output_dim ==1, "ep in 1D only (for now!)"

        K = kern.K(X)

        if self._ep_approximation is None:

            #if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation = self.expectation_propagation(K, Y, likelihood, Y_metadata)
        else:
            #if we've already run EP, just use the existing approximation stored in self._ep_approximation
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation

        Wi, LW, LWi, W_logdet = pdinv(K + np.diag(1./tau_tilde))

        alpha, _ = dpotrs(LW, mu_tilde, lower=1)

        log_marginal =  0.5*(-num_data * log_2_pi - W_logdet - np.sum(alpha * mu_tilde)) # TODO: add log Z_hat??

        dL_dK = 0.5 * (tdot(alpha[:,None]) - Wi)

        dL_dthetaL = np.zeros(likelihood.size)#TODO: derivatives of the likelihood parameters

        return Posterior(woodbury_inv=Wi, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}

    def expectation_propagation(self, K, Y, likelihood, Y_metadata):

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"


        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(num_data)
        Sigma = K.copy()

        #Initial values - Marginal moments
        Z_hat = np.empty(num_data,dtype=np.float64)
        mu_hat = np.empty(num_data,dtype=np.float64)
        sigma2_hat = np.empty(num_data,dtype=np.float64)

        #initial values - Gaussian factors
        if self.old_mutilde is None:
            tau_tilde, mu_tilde, v_tilde = np.zeros((3, num_data))
        else:
            assert old_mutilde.size == num_data, "data size mis-match: did you change the data? try resetting!"
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
