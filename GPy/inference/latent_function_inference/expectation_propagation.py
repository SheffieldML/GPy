# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
import itertools
from ...util.linalg import jitchol, DSYR, dtrtrs, dtrtri, pdinv, dpotrs, tdot, symmetrify
from paramz import ObsAr
from . import ExactGaussianInference, VarDTC
from ...util import diag
from .posterior import PosteriorEP as Posterior

log_2_pi = np.log(2*np.pi)

class EPBase(object):
    def __init__(self, epsilon=1e-6, eta=1., delta=1., always_reset=False, max_iters=np.inf, ep_mode="alternated", parallel_updates=False):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006.

        :param epsilon: Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :type epsilon: float
        :param eta: parameter for fractional EP updates.
        :type eta: float64
        :param delta: damping EP updates factor.
        :type delta: float64
        :param always_reset: setting to always reset the approximation at the beginning of every inference call.
        :type always_reest: boolean
        :max_iters: int
        :ep_mode: string. It can be "nested" (EP is run every time the Hyperparameters change) or "alternated" (It runs EP at the beginning and then optimize the Hyperparameters).
        :parallel_updates: boolean. If true, updates of the parameters of the sites in parallel
        """
        super(EPBase, self).__init__()
        self.always_reset = always_reset
        self.epsilon, self.eta, self.delta, self.max_iters = epsilon, eta, delta, max_iters
        self.ep_mode = ep_mode
        self.parallel_updates = parallel_updates
        self.reset()

    def reset(self):
        self.old_mutilde, self.old_vtilde = None, None
        self.ga_approx_old = None
        self._ep_approximation = None

    def on_optimization_start(self):
        self._ep_approximation = None

    def on_optimization_end(self):
        # TODO: update approximation in the end as well? Maybe even with a switch?
        pass

    def __setstate__(self, state):
        super(EPBase, self).__setstate__(state[0])
        self.epsilon, self.eta, self.delta = state[1]
        self.reset()

    def __getstate__(self):
        return [super(EPBase, self).__getstate__() , [self.epsilon, self.eta, self.delta]]

class EP(EPBase, ExactGaussianInference):
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, precision=None, K=None):
        if self.always_reset:
            self.reset()

        num_data, output_dim = Y.shape
        assert output_dim == 1, "ep in 1D only (for now!)"

        if K is None:
            K = kern.K(X)

        if self.ep_mode=="nested":
            #Force EP at each step of the optimization
            self._ep_approximation = None
            mu, Sigma, mu_tilde, tau_tilde, log_Z_tilde = self._ep_approximation = self.expectation_propagation(K, Y, likelihood, Y_metadata)
        elif self.ep_mode=="alternated":
            if getattr(self, '_ep_approximation', None) is None:
                #if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
                mu, Sigma, mu_tilde, tau_tilde, log_Z_tilde = self._ep_approximation = self.expectation_propagation(K, Y, likelihood, Y_metadata)
            else:
                #if we've already run EP, just use the existing approximation stored in self._ep_approximation
                mu, Sigma, mu_tilde, tau_tilde, log_Z_tilde = self._ep_approximation
        else:
            raise ValueError("ep_mode value not valid")

        v_tilde = mu_tilde * tau_tilde
        return self._inference(K, tau_tilde, v_tilde, likelihood, Y_metadata=Y_metadata,  Z_tilde=log_Z_tilde.sum())

    def expectation_propagation(self, K, Y, likelihood, Y_metadata):

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"

        # Makes computing the sign quicker if we work with numpy arrays rather
        # than ObsArrays
        Y = Y.values.copy()

        #Initial values - Marginal moments, cavity params, gaussian approximation params and posterior params
        marg_moments = marginalMoments(num_data)
        cav_params = cavityParams(num_data)
        ga_approx, post_params = self._init_approximations(K, num_data)

        #Approximation
        tau_diff = self.epsilon + 1.
        v_diff = self.epsilon + 1.
        
        iterations = 0
        while ((tau_diff > self.epsilon) or (v_diff > self.epsilon)) and (iterations < self.max_iters):
            self._update_cavity_params(num_data, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata)

            #(re) compute Sigma and mu using full Cholesky decompy
            post_params = self._ep_compute_posterior(K, ga_approx.tau, ga_approx.v)

            #monitor convergence
            if iterations > 0:
                tau_diff = np.mean(np.square(ga_approx.tau-self.ga_approx_old.tau))
                v_diff = np.mean(np.square(ga_approx.v-self.ga_approx_old.v))
            self.ga_approx_old = gaussianApproximation(ga_approx.mu.copy(), ga_approx.v.copy(), ga_approx.tau.copy())
            iterations += 1

        ga_approx.mu = ga_approx.v/ga_approx.tau

        # Z_tilde after removing the terms that can lead to infinite terms due to tau_tilde close to zero.
        # This terms cancel with the coreresponding terms in the marginal loglikelihood
        log_Z_tilde = self._log_Z_tilde(marg_moments, ga_approx, cav_params)
                         # - 0.5*np.log(tau_tilde) + 0.5*(v_tilde*v_tilde*1./tau_tilde)

        return post_params.mu, post_params.Sigma, ga_approx.mu, ga_approx.tau, log_Z_tilde
    
    def _log_Z_tilde(self, marg_moments, ga_approx, cav_params):
        return (np.log(marg_moments.Z_hat) + 0.5*np.log(2*np.pi) + 0.5*np.log(1+ga_approx.tau/cav_params.tau) - 0.5 * ((ga_approx.v)**2 * 1./(cav_params.tau + ga_approx.tau)) 
                + 0.5*(cav_params.v * ( ( (ga_approx.tau/cav_params.tau) * cav_params.v - 2.0 * ga_approx.v ) * 1./(cav_params.tau + ga_approx.tau))))
    
    def _update_cavity_params(self, num_data, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata, update_order=None):
            if update_order is None:
                update_order = np.random.permutation(num_data)
            for i in update_order:
                #Cavity distribution parameters
                cav_params.tau[i] = 1./post_params.Sigma[i,i] - self.eta*ga_approx.tau[i]
                cav_params.v[i] = post_params.mu[i]/post_params.Sigma[i,i] - self.eta*ga_approx.v[i]
                if Y_metadata is not None:
                    # Pick out the relavent metadata for Yi
                    Y_metadata_i = {}
                    for key in Y_metadata.keys():
                        Y_metadata_i[key] = Y_metadata[key][i, :]
                else:
                    Y_metadata_i = None
                #Marginal moments
                marg_moments.Z_hat[i], marg_moments.mu_hat[i], marg_moments.sigma2_hat[i] = likelihood.moments_match_ep(Y[i], cav_params.tau[i], cav_params.v[i], Y_metadata_i=Y_metadata_i)
                
                #Site parameters update
                delta_tau = self.delta/self.eta*(1./marg_moments.sigma2_hat[i] - 1./post_params.Sigma[i,i])
                delta_v = self.delta/self.eta*(marg_moments.mu_hat[i]/marg_moments.sigma2_hat[i] - post_params.mu[i]/post_params.Sigma[i,i])
                tau_tilde_prev = ga_approx.tau[i]
                ga_approx.tau[i] += delta_tau

                # Enforce positivity of tau_tilde. Even though this is guaranteed for logconcave sites, it is still possible
                # to get negative values due to numerical errors. Moreover, the value of tau_tilde should be positive in order to
                # update the marginal likelihood without inestability issues.
                if ga_approx.tau[i] < np.finfo(float).eps:
                    ga_approx.tau[i] = np.finfo(float).eps
                    delta_tau = ga_approx.tau[i] - tau_tilde_prev
                ga_approx.v[i] += delta_v

                if self.parallel_updates == False:
                    #Posterior distribution parameters update
                    ci = delta_tau/(1.+ delta_tau*post_params.Sigma[i,i])
                    DSYR(post_params.Sigma, post_params.Sigma[:,i].copy(), -ci)
                    post_params.mu = np.dot(post_params.Sigma, ga_approx.v)
                    
    def _init_approximations(self, K, num_data):
        #initial values - Gaussian factors
        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        if self.ga_approx_old is None:
            mu_tilde, v_tilde, tau_tilde = np.zeros((3, num_data))
            ga_approx = gaussianApproximation(mu_tilde, v_tilde, tau_tilde)
            Sigma = K.copy()
            diag.add(Sigma, 1e-7)
            mu = np.zeros(num_data)
            post_params = posteriorParams(mu, Sigma)
        else:
            assert self.ga_approx_old.mu.size == num_data, "data size mis-match: did you change the data? try resetting!"
            ga_approx = gaussianApproximation(self.ga_approx_old.mu, self.ga_approx_old.v)
            post_params = self._ep_compute_posterior(K, ga_approx.tau, ga_approx.v)
            diag.add(post_params.Sigma, 1e-7)
            # TODO: Check the log-marginal under both conditions and choose the best one
        return (ga_approx, post_params)
         
    def _ep_compute_posterior(self, K, tau_tilde, v_tilde):
        num_data = len(tau_tilde)
        tau_tilde_root = np.sqrt(tau_tilde)
        Sroot_tilde_K = tau_tilde_root[:,None] * K
        B = np.eye(num_data) + Sroot_tilde_K * tau_tilde_root[None,:]
        L = jitchol(B)
        V, _ = dtrtrs(L, Sroot_tilde_K, lower=1)
        Sigma = K - np.dot(V.T,V) #K - KS^(1/2)BS^(1/2)K = (K^(-1) + \Sigma^(-1))^(-1)
        mu = np.dot(Sigma,v_tilde)
        return posteriorParams(mu, Sigma, L)

    def _ep_marginal(self, K, tau_tilde, v_tilde, Z_tilde):
        post_params = self._ep_compute_posterior(K, tau_tilde, v_tilde)

        # Gaussian log marginal excluding terms that can go to infinity due to arbitrarily small tau_tilde.
        # These terms cancel out with the terms excluded from Z_tilde
        B_logdet = np.sum(2.0*np.log(np.diag(post_params.L)))
        log_marginal =  0.5*(-len(tau_tilde) * log_2_pi - B_logdet + np.sum(v_tilde * np.dot(post_params.Sigma,v_tilde)))
        log_marginal += Z_tilde

        return log_marginal, post_params.mu, post_params.Sigma, post_params.L

    def _inference(self, K, tau_tilde, v_tilde, likelihood, Z_tilde, Y_metadata=None):
        log_marginal, mu, Sigma, L = self._ep_marginal(K, tau_tilde, v_tilde, Z_tilde)

        tau_tilde_root = np.sqrt(tau_tilde)
        Sroot_tilde_K = tau_tilde_root[:,None] * K

        aux_alpha , _ = dpotrs(L, np.dot(Sroot_tilde_K, v_tilde), lower=1)
        alpha = (v_tilde - tau_tilde_root * aux_alpha)[:,None] #(K + Sigma^(\tilde))^(-1) /mu^(/tilde)
        LWi, _ = dtrtrs(L, np.diag(tau_tilde_root), lower=1)
        Wi = np.dot(LWi.T,LWi)
        symmetrify(Wi) #(K + Sigma^(\tilde))^(-1)

        dL_dK = 0.5 * (tdot(alpha) - Wi)
        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return Posterior(woodbury_inv=Wi, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}


class EPDTC(EPBase, VarDTC):
    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, Lm=None, dL_dKmm=None, psi0=None, psi1=None, psi2=None):
        if self.always_reset:
            self.reset()

        num_data, output_dim = Y.shape
        assert output_dim == 1, "ep in 1D only (for now!)"

        if Lm is None:
            Kmm = kern.K(Z)
            Lm = jitchol(Kmm)

        if psi1 is None:
            try:
                Kmn = kern.K(Z, X)
            except TypeError:
                Kmn = kern.psi1(Z, X).T
        else:
            Kmn = psi1.T

        if self.ep_mode=="nested":
            #Force EP at each step of the optimization
            self._ep_approximation = None
            mu, Sigma_diag, mu_tilde, tau_tilde, log_Z_tilde = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
        elif self.ep_mode=="alternated":
            if getattr(self, '_ep_approximation', None) is None:
                #if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
                mu, Sigma_diag, mu_tilde, tau_tilde, log_Z_tilde = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
            else:
                #if we've already run EP, just use the existing approximation stored in self._ep_approximation
                mu, Sigma_diag, mu_tilde, tau_tilde, log_Z_tilde = self._ep_approximation
        else:
            raise ValueError("ep_mode value not valid")

        return super(EPDTC, self).inference(kern, X, Z, likelihood, mu_tilde,
                                            mean_function=mean_function,
                                            Y_metadata=Y_metadata,
                                            precision=tau_tilde,
                                            Lm=Lm, dL_dKmm=dL_dKmm,
                                            psi0=psi0, psi1=psi1, psi2=psi2, Z_tilde=log_Z_tilde.sum())


    def expectation_propagation(self, Kmm, Kmn, Y, likelihood, Y_metadata):

        num_data, output_dim = Y.shape
        assert output_dim == 1, "This EP methods only works for 1D outputs"

        # Makes computing the sign quicker if we work with numpy arrays rather
        # than ObsArrays
        Y = Y.values.copy()

        #Initial values - Marginal moments
        Z_hat = np.zeros(num_data,dtype=np.float64)
        mu_hat = np.zeros(num_data,dtype=np.float64)
        sigma2_hat = np.zeros(num_data,dtype=np.float64)

        tau = np.empty(num_data,dtype=np.float64)
        v = np.empty(num_data,dtype=np.float64)

        #initial values - Gaussian factors
        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        LLT0 = Kmm.copy()
        Lm = jitchol(LLT0) #K_m = L_m L_m^\top
        Vm,info = dtrtrs(Lm,Kmn,lower=1)
        # Lmi = dtrtri(Lm)
        # Kmmi = np.dot(Lmi.T,Lmi)
        # KmmiKmn = np.dot(Kmmi,Kmn)
        # Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        Qnn_diag = np.sum(Vm*Vm,-2) #diag(Knm Kmm^(-1) Kmn)
        #diag.add(LLT0, 1e-8)
        if self.old_mutilde is None:
            #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
            LLT = LLT0.copy() #Sigma = K.copy()
            mu = np.zeros(num_data)
            Sigma_diag = Qnn_diag.copy() + 1e-8
            tau_tilde, mu_tilde, v_tilde = np.zeros((3, num_data))
        else:
            assert self.old_mutilde.size == num_data, "data size mis-match: did you change the data? try resetting!"
            mu_tilde, v_tilde = self.old_mutilde, self.old_vtilde
            tau_tilde = v_tilde/mu_tilde
            mu, Sigma_diag, LLT = self._ep_compute_posterior(LLT0, Kmn, tau_tilde, v_tilde)
            Sigma_diag += 1e-8
            # TODO: Check the log-marginal under both conditions and choose the best one

        #Approximation
        tau_diff = self.epsilon + 1.
        v_diff = self.epsilon + 1.
        tau_tilde_old = np.nan
        v_tilde_old = np.nan
        iterations = 0
        while  ((tau_diff > self.epsilon) or (v_diff > self.epsilon)) and (iterations < self.max_iters):
            update_order = np.random.permutation(num_data)
            for i in update_order:
                #Cavity distribution parameters
                tau[i] = 1./Sigma_diag[i] - self.eta*tau_tilde[i]
                v[i] = mu[i]/Sigma_diag[i] - self.eta*v_tilde[i]
                if Y_metadata is not None:
                    # Pick out the relavent metadata for Yi
                    Y_metadata_i = {}
                    for key in Y_metadata.keys():
                        Y_metadata_i[key] = Y_metadata[key][i, :]
                else:
                    Y_metadata_i = None

                #Marginal moments
                Z_hat[i], mu_hat[i], sigma2_hat[i] = likelihood.moments_match_ep(Y[i], tau[i], v[i], Y_metadata_i=Y_metadata_i)
                #Site parameters update
                delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                tau_tilde_prev = tau_tilde[i]
                tau_tilde[i] += delta_tau

                # Enforce positivity of tau_tilde. Even though this is guaranteed for logconcave sites, it is still possible
                # to get negative values due to numerical errors. Moreover, the value of tau_tilde should be positive in order to
                # update the marginal likelihood without inestability issues.
                if tau_tilde[i] < np.finfo(float).eps:
                    tau_tilde[i] = np.finfo(float).eps
                    delta_tau = tau_tilde[i] - tau_tilde_prev
                v_tilde[i] += delta_v

                #Posterior distribution parameters update
                if self.parallel_updates == False:
                    #DSYR(Sigma, Sigma[:,i].copy(), -delta_tau/(1.+ delta_tau*Sigma[i,i]))
                    DSYR(LLT,Kmn[:,i].copy(),delta_tau)
                    L = jitchol(LLT)
                    V,info = dtrtrs(L,Kmn,lower=1)
                    Sigma_diag = np.maximum(np.sum(V*V,-2), np.finfo(float).eps)  #diag(K_nm (L L^\top)^(-1)) K_mn
                    si = np.sum(V.T*V[:,i],-1) #(V V^\top)[:,i]
                    mu += (delta_v-delta_tau*mu[i])*si
                    #mu = np.dot(Sigma, v_tilde)

            #(re) compute Sigma, Sigma_diag and mu using full Cholesky decompy
            mu, Sigma_diag, LLT = self._ep_compute_posterior(LLT0, Kmn, tau_tilde, v_tilde)
            Sigma_diag = np.maximum(Sigma_diag, np.finfo(float).eps)

            #monitor convergence
            if iterations>0:
                tau_diff = np.mean(np.square(tau_tilde-tau_tilde_old))
                v_diff = np.mean(np.square(v_tilde-v_tilde_old))
            tau_tilde_old = tau_tilde.copy()
            v_tilde_old = v_tilde.copy()
            iterations += 1

        mu_tilde = v_tilde/tau_tilde
        mu_cav = v/tau
        sigma2_sigma2tilde = 1./tau + 1./tau_tilde

        log_Z_tilde = (np.log(Z_hat) + 0.5*np.log(2*np.pi) + 0.5*np.log(sigma2_sigma2tilde)
                         + 0.5*((mu_cav - mu_tilde)**2) / (sigma2_sigma2tilde))

        self.old_mutilde = mu_tilde
        self.old_vtilde = v_tilde

        return mu, Sigma_diag, ObsAr(mu_tilde[:,None]), tau_tilde, log_Z_tilde

    def _ep_compute_posterior(self, LLT0, Kmn, tau_tilde, v_tilde):
        LLT = LLT0 + np.dot(Kmn*tau_tilde[None,:],Kmn.T)
        L = jitchol(LLT)
        V, _ = dtrtrs(L,Kmn,lower=1)
        #Sigma_diag = np.sum(V*V,-2)
        #Knmv_tilde = np.dot(Kmn,v_tilde)
        #mu = np.dot(V2.T,Knmv_tilde)
        Sigma = np.dot(V.T,V)
        mu = np.dot(Sigma,v_tilde)
        Sigma_diag = np.diag(Sigma).copy()

        return (mu, Sigma_diag, LLT)

#Four wrapper classes to help modularisation of different EP versions
class marginalMoments(object):
    def __init__(self, num_data):
        #Initial values - Marginal moments
        self.Z_hat = np.empty(num_data,dtype=np.float64)
        self.mu_hat = np.empty(num_data,dtype=np.float64)
        self.sigma2_hat = np.empty(num_data,dtype=np.float64)

class cavityParams(object):
    def __init__(self, num_data):
        self.tau = np.empty(num_data,dtype=np.float64)
        self.v = np.empty(num_data,dtype=np.float64)
        
class gaussianApproximation(object):
    def __init__(self, mu, v, tau=None):
        self.mu = mu
        self.v = v
        self.tau = mu / v if tau is None else tau
        
class posteriorParams(object):
    def __init__(self, mu=None, Sigma=None, L=None):
        self.mu = mu 
        self.Sigma = Sigma
        self.L = L
