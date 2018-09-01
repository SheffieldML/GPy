# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from ...util.linalg import jitchol, DSYR, dtrtrs, dtrtri, pdinv, dpotrs, tdot, symmetrify
from paramz import ObsAr
from . import ExactGaussianInference, VarDTC
from ...util import diag
from .posterior import PosteriorEP as Posterior
from ...likelihoods import Gaussian
from . import LatentFunctionInference

log_2_pi = np.log(2*np.pi)


#Four wrapper classes to help modularisation of different EP versions
class marginalMoments(object):
    def __init__(self, num_data):
        self.Z_hat = np.empty(num_data,dtype=np.float64)
        self.mu_hat = np.empty(num_data,dtype=np.float64)
        self.sigma2_hat = np.empty(num_data,dtype=np.float64)


class cavityParams(object):
    def __init__(self, num_data):
        self.tau = np.empty(num_data,dtype=np.float64)
        self.v = np.empty(num_data,dtype=np.float64)
    def _update_i(self, eta, ga_approx, post_params, i):
        self.tau[i] = 1./post_params.Sigma_diag[i] - eta*ga_approx.tau[i]
        self.v[i] = post_params.mu[i]/post_params.Sigma_diag[i] - eta*ga_approx.v[i]
    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        return {"tau": self.tau.tolist(), "v": self.v.tolist()}
    @staticmethod
    def from_dict(input_dict):
        c = cavityParams(len(input_dict["tau"]))
        c.tau = np.array(input_dict["tau"])
        c.v = np.array(input_dict["v"])
        return c


class gaussianApproximation(object):
    def __init__(self, v, tau):
        self.tau = tau
        self.v = v
    def _update_i(self, eta, delta, post_params, marg_moments, i):
        #Site parameters update
        delta_tau = delta/eta*(1./marg_moments.sigma2_hat[i] - 1./post_params.Sigma_diag[i])
        delta_v = delta/eta*(marg_moments.mu_hat[i]/marg_moments.sigma2_hat[i] - post_params.mu[i]/post_params.Sigma_diag[i])
        tau_tilde_prev = self.tau[i]
        self.tau[i] += delta_tau

        # Enforce positivity of tau_tilde. Even though this is guaranteed for logconcave sites, it is still possible
        # to get negative values due to numerical errors. Moreover, the value of tau_tilde should be positive in order to
        # update the marginal likelihood without runnint into instabilities issues.
        if self.tau[i] < np.finfo(float).eps:
            self.tau[i] = np.finfo(float).eps
            delta_tau = self.tau[i] - tau_tilde_prev

        self.v[i] += delta_v

        return (delta_tau, delta_v)
    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        return {"tau": self.tau.tolist(), "v": self.v.tolist()}
    @staticmethod
    def from_dict(input_dict):
        return gaussianApproximation(np.array(input_dict["v"]), np.array(input_dict["tau"]))


class posteriorParamsBase(object):
    def __init__(self, mu, Sigma_diag):
        self.mu = mu
        self.Sigma_diag = Sigma_diag
    def _update_rank1(self, *arg):
        pass

    def _recompute(self, *arg):
        pass

class posteriorParams(posteriorParamsBase):
    def __init__(self, mu, Sigma, L=None):
        self.Sigma = Sigma
        self.L = L
        Sigma_diag = np.diag(self.Sigma)
        super(posteriorParams, self).__init__(mu, Sigma_diag)

    def _update_rank1(self, delta_tau, delta_v, ga_approx, i):
        si = self.Sigma[i,:].copy()
        ci = delta_tau/(1.+ delta_tau*si[i])
        self.mu = self.mu - (ci*(self.mu[i]+si[i]*delta_v)-delta_v) * si
        DSYR(self.Sigma, si, -ci)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        #TODO: Implement a more memory efficient variant
        if self.L is None:
            return { "mu": self.mu.tolist(), "Sigma": self.Sigma.tolist()}
        else:
            return { "mu": self.mu.tolist(), "Sigma": self.Sigma.tolist(), "L": self.L.tolist()}

    @staticmethod
    def from_dict(input_dict):
        if "L" in input_dict:
            return posteriorParams(np.array(input_dict["mu"]), np.array(input_dict["Sigma"]), np.array(input_dict["L"]))
        else:
            return posteriorParams(np.array(input_dict["mu"]), np.array(input_dict["Sigma"]))

    @staticmethod
    def _recompute(mean_prior, K, ga_approx):
        num_data = len(ga_approx.tau)
        tau_tilde_root = np.sqrt(ga_approx.tau)
        Sroot_tilde_K = tau_tilde_root[:,None] * K
        B = np.eye(num_data) + Sroot_tilde_K * tau_tilde_root[None,:]
        L = jitchol(B)
        V, _ = dtrtrs(L, Sroot_tilde_K, lower=1)
        Sigma = K - np.dot(V.T,V) #K - KS^(1/2)BS^(1/2)K = (K^(-1) + \Sigma^(-1))^(-1)

        aux_alpha , _ = dpotrs(L, tau_tilde_root * (np.dot(K, ga_approx.v) + mean_prior), lower=1)
        alpha = ga_approx.v - tau_tilde_root * aux_alpha #(K + Sigma^(\tilde))^(-1) (/mu^(/tilde) - /mu_p)
        mu = np.dot(K, alpha) + mean_prior

        return posteriorParams(mu=mu, Sigma=Sigma, L=L)

class posteriorParamsDTC(posteriorParamsBase):
    def __init__(self, mu, Sigma_diag):
        super(posteriorParamsDTC, self).__init__(mu, Sigma_diag)

    def _update_rank1(self, LLT, Kmn, delta_v, delta_tau, i):
        #DSYR(Sigma, Sigma[:,i].copy(), -delta_tau/(1.+ delta_tau*Sigma[i,i]))
        DSYR(LLT,Kmn[:,i].copy(),delta_tau)
        L = jitchol(LLT)
        V,info = dtrtrs(L,Kmn,lower=1)
        self.Sigma_diag = np.maximum(np.sum(V*V,-2), np.finfo(float).eps)  #diag(K_nm (L L^\top)^(-1)) K_mn
        si = np.sum(V.T*V[:,i],-1) #(V V^\top)[:,i]
        self.mu += (delta_v-delta_tau*self.mu[i])*si
        #mu = np.dot(Sigma, v_tilde)

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        return { "mu": self.mu.tolist(), "Sigma_diag": self.Sigma_diag.tolist()}

    @staticmethod
    def from_dict(input_dict):
        return posteriorParamsDTC(np.array(input_dict["mu"]), np.array(input_dict["Sigma_diag"]))

    @staticmethod
    def _recompute(LLT0, Kmn, ga_approx):
        LLT = LLT0 + np.dot(Kmn*ga_approx.tau[None,:],Kmn.T)
        L = jitchol(LLT)
        V, _ = dtrtrs(L,Kmn,lower=1)
        #Sigma_diag = np.sum(V*V,-2)
        #Knmv_tilde = np.dot(Kmn,v_tilde)
        #mu = np.dot(V2.T,Knmv_tilde)
        Sigma = np.dot(V.T,V)
        mu = np.dot(Sigma, ga_approx.v)
        Sigma_diag = np.diag(Sigma).copy()
        return posteriorParamsDTC(mu, Sigma_diag), LLT

class EPBase(object):
    def __init__(self, epsilon=1e-6, eta=1., delta=1., always_reset=False, max_iters=np.inf, ep_mode="alternated", parallel_updates=False, loading=False):
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
        :loading: boolean. If True, prevents the EP parameters to change. Hack used when loading a serialized model
        """
        super(EPBase, self).__init__()

        self.always_reset = always_reset
        self.epsilon, self.eta, self.delta, self.max_iters = epsilon, eta, delta, max_iters
        self.ep_mode = ep_mode
        self.parallel_updates = parallel_updates
        #FIXME: Hack for serialiation. If True, prevents the EP parameters to change when loading a serialized model
        self.loading = loading
        self.reset()

    def reset(self):
        self.ga_approx_old = None
        self._ep_approximation = None

    def on_optimization_start(self):
        self._ep_approximation = None

    def on_optimization_end(self):
        # TODO: update approximation in the end as well? Maybe even with a switch?
        pass

    def _stop_criteria(self, ga_approx):
        tau_diff = np.mean(np.square(ga_approx.tau-self.ga_approx_old.tau))
        v_diff = np.mean(np.square(ga_approx.v-self.ga_approx_old.v))
        return ((tau_diff < self.epsilon) and (v_diff < self.epsilon))

    def __setstate__(self, state):
        super(EPBase, self).__setstate__(state[0])
        self.epsilon, self.eta, self.delta = state[1]
        self.reset()

    def __getstate__(self):
        return [super(EPBase, self).__getstate__() , [self.epsilon, self.eta, self.delta]]

    def _save_to_input_dict(self):
        input_dict = super(EPBase, self)._save_to_input_dict()
        input_dict["epsilon"]=self.epsilon
        input_dict["eta"]=self.eta
        input_dict["delta"]=self.delta
        input_dict["always_reset"]=self.always_reset
        input_dict["max_iters"]=self.max_iters
        input_dict["ep_mode"]=self.ep_mode
        input_dict["parallel_updates"]=self.parallel_updates
        input_dict["loading"]=True
        return input_dict

class EP(EPBase, ExactGaussianInference):
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, precision=None, K=None):
        if self.always_reset and not self.loading:
            self.reset()

        num_data, output_dim = Y.shape
        assert output_dim == 1, "ep in 1D only (for now!)"

        if mean_function is None:
            mean_prior = np.zeros(X.shape[0])
        else:
            mean_prior = mean_function.f(X).flatten()

        if K is None:
            K = kern.K(X)

        if self.ep_mode=="nested" and not self.loading:
            #Force EP at each step of the optimization
            self._ep_approximation = None
            post_params, ga_approx, cav_params, log_Z_tilde = self._ep_approximation = self.expectation_propagation(mean_prior, K, Y, likelihood, Y_metadata)
        elif self.ep_mode=="alternated" or self.loading:
            if getattr(self, '_ep_approximation', None) is None:
                #if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
                post_params, ga_approx, cav_params, log_Z_tilde = self._ep_approximation = self.expectation_propagation(mean_prior, K, Y, likelihood, Y_metadata)
            else:
                #if we've already run EP, just use the existing approximation stored in self._ep_approximation
                post_params, ga_approx, cav_params, log_Z_tilde = self._ep_approximation
        else:
            raise ValueError("ep_mode value not valid")

        self.loading = False

        return self._inference(Y, mean_prior, K, ga_approx, cav_params, likelihood, Y_metadata=Y_metadata,  Z_tilde=log_Z_tilde)

    def expectation_propagation(self, mean_prior, K, Y, likelihood, Y_metadata):

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"

        # Makes computing the sign quicker if we work with numpy arrays rather
        # than ObsArrays
        Y = Y.values.copy()

        #Initial values - Marginal moments, cavity params, gaussian approximation params and posterior params
        marg_moments = marginalMoments(num_data)
        cav_params = cavityParams(num_data)
        ga_approx, post_params = self._init_approximations(mean_prior, K, num_data)

        #Approximation
        stop = False
        iterations = 0
        while not stop and (iterations < self.max_iters):
            self._local_updates(num_data, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata)

            #(re) compute Sigma and mu using full Cholesky decompy
            post_params = posteriorParams._recompute(mean_prior, K, ga_approx)

            #monitor convergence
            if iterations > 0:
                stop = self._stop_criteria(ga_approx)
            self.ga_approx_old = gaussianApproximation(ga_approx.v.copy(), ga_approx.tau.copy())
            iterations += 1

        log_Z_tilde = self._log_Z_tilde(marg_moments, ga_approx, cav_params)

        return (post_params, ga_approx, cav_params, log_Z_tilde)

    def _init_approximations(self, mean_prior, K, num_data):
        #initial values - Gaussian factors
        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        if self.ga_approx_old is None:
            v_tilde, tau_tilde = np.zeros((2, num_data))
            ga_approx = gaussianApproximation(v_tilde, tau_tilde)
            Sigma = K.copy()
            diag.add(Sigma, 1e-7)
            mu = mean_prior
            post_params = posteriorParams(mu, Sigma)
        else:
            assert self.ga_approx_old.v.size == num_data, "data size mis-match: did you change the data? try resetting!"
            ga_approx = gaussianApproximation(self.ga_approx_old.v, self.ga_approx_old.tau)
            post_params = posteriorParams._recompute(mean_prior, K, ga_approx)
            diag.add(post_params.Sigma, 1e-7)
            # TODO: Check the log-marginal under both conditions and choose the best one
        return (ga_approx, post_params)

    def _local_updates(self, num_data, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata, update_order=None):
            if update_order is None:
                update_order = np.random.permutation(num_data)
            for i in update_order:
                #Cavity distribution parameters
                cav_params._update_i(self.eta, ga_approx, post_params, i)

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
                delta_tau, delta_v = ga_approx._update_i(self.eta, self.delta, post_params, marg_moments, i)

                if self.parallel_updates == False:
                    post_params._update_rank1(delta_tau, delta_v, ga_approx, i)

    def _log_Z_tilde(self, marg_moments, ga_approx, cav_params):
        # Z_tilde after removing the terms that can lead to infinite terms due to tau_tilde close to zero.
        # This terms cancel with the coreresponding terms in the marginal loglikelihood
        return np.sum((
                np.log(marg_moments.Z_hat)
                + 0.5*np.log(2*np.pi) + 0.5*np.log(1+ga_approx.tau/cav_params.tau)
                - 0.5 * ((ga_approx.v)**2 * 1./(cav_params.tau + ga_approx.tau))
                + 0.5*(cav_params.v * ( ( (ga_approx.tau/cav_params.tau) * cav_params.v - 2.0 * ga_approx.v ) * 1./(cav_params.tau + ga_approx.tau)))
                ))

    def _ep_marginal(self, mean_prior, K, ga_approx, Z_tilde):
        post_params = posteriorParams._recompute(mean_prior, K, ga_approx)
        # Gaussian log marginal excluding terms that can go to infinity due to arbitrarily small tau_tilde.
        # These terms cancel out with the terms excluded from Z_tilde
        B_logdet = np.sum(2.0*np.log(np.diag(post_params.L)))
        S_mean_prior = ga_approx.tau * mean_prior
        v_centered = ga_approx.v - S_mean_prior
        log_marginal =  0.5*(
                        -len(ga_approx.tau) * log_2_pi - B_logdet
                        + np.sum(v_centered * np.dot(post_params.Sigma, v_centered))
                        - np.dot(mean_prior, (S_mean_prior - 2*ga_approx.v))
                        )
        log_marginal += Z_tilde

        return log_marginal, post_params

    def _inference(self, Y, mean_prior, K, ga_approx, cav_params, likelihood, Z_tilde, Y_metadata=None):
        log_marginal, post_params = self._ep_marginal(mean_prior, K, ga_approx, Z_tilde)

        tau_tilde_root = np.sqrt(ga_approx.tau)
        Sroot_tilde_K = tau_tilde_root[:,None] * K


        aux_alpha , _ = dpotrs(post_params.L, tau_tilde_root * (np.dot(K, ga_approx.v) +  mean_prior), lower=1)
        alpha = (ga_approx.v - tau_tilde_root * aux_alpha)[:,None] #(K + Sigma^(\tilde))^(-1) (/mu^(/tilde) -  /mu_p)

        LWi, _ = dtrtrs(post_params.L, np.diag(tau_tilde_root), lower=1)
        Wi = np.dot(LWi.T,LWi)
        symmetrify(Wi) #(K + Sigma^(\tilde))^(-1)

        dL_dK = 0.5 * (tdot(alpha) - Wi)
        dL_dthetaL = likelihood.ep_gradients(Y, cav_params.tau, cav_params.v, np.diag(dL_dK), Y_metadata=Y_metadata, quad_mode='gh')
        return Posterior(woodbury_inv=Wi, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(EP, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.expectation_propagation.EP"
        if self.ga_approx_old is not  None:
            input_dict["ga_approx_old"] = self.ga_approx_old.to_dict()
        if self._ep_approximation is not  None:
            input_dict["_ep_approximation"] = {}
            input_dict["_ep_approximation"]["post_params"] = self._ep_approximation[0].to_dict()
            input_dict["_ep_approximation"]["ga_approx"] = self._ep_approximation[1].to_dict()
            input_dict["_ep_approximation"]["cav_params"] = self._ep_approximation[2].to_dict()
            input_dict["_ep_approximation"]["log_Z_tilde"] = self._ep_approximation[3].tolist()

        return input_dict

    @staticmethod
    def _build_from_input_dict(inference_class, input_dict):
        ga_approx_old = input_dict.pop('ga_approx_old', None)
        if ga_approx_old is not None:
            ga_approx_old = gaussianApproximation.from_dict(ga_approx_old)
        _ep_approximation_dict = input_dict.pop('_ep_approximation', None)
        _ep_approximation = []
        if _ep_approximation is not None:
            _ep_approximation.append(posteriorParams.from_dict(_ep_approximation_dict["post_params"]))
            _ep_approximation.append(gaussianApproximation.from_dict(_ep_approximation_dict["ga_approx"]))
            _ep_approximation.append(cavityParams.from_dict(_ep_approximation_dict["cav_params"]))
            _ep_approximation.append(np.array(_ep_approximation_dict["log_Z_tilde"]))
        ee = EP(**input_dict)
        ee.ga_approx_old = ga_approx_old
        ee._ep_approximation = _ep_approximation
        return ee

class EPDTC(EPBase, VarDTC):
    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, Lm=None, dL_dKmm=None, psi0=None, psi1=None, psi2=None):
        if self.always_reset and not self.loading:
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

        if self.ep_mode=="nested" and not self.loading:
            #Force EP at each step of the optimization
            self._ep_approximation = None
            post_params, ga_approx, log_Z_tilde = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
        elif self.ep_mode=="alternated" or self.loading:
            if getattr(self, '_ep_approximation', None) is None:
                #if we don't yet have the results of runnign EP, run EP and store the computed factors in self._ep_approximation
                post_params, ga_approx, log_Z_tilde = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
            else:
                #if we've already run EP, just use the existing approximation stored in self._ep_approximation
                post_params, ga_approx, log_Z_tilde = self._ep_approximation
        else:
            raise ValueError("ep_mode value not valid")

        self.loading = False

        mu_tilde = ga_approx.v / ga_approx.tau.astype(float)

        return super(EPDTC, self).inference(kern, X, Z, likelihood, ObsAr(mu_tilde[:,None]),
                                            mean_function=mean_function,
                                            Y_metadata=Y_metadata,
                                            precision=ga_approx.tau,
                                            Lm=Lm, dL_dKmm=dL_dKmm,
                                            psi0=psi0, psi1=psi1, psi2=psi2, Z_tilde=log_Z_tilde)

    def expectation_propagation(self, Kmm, Kmn, Y, likelihood, Y_metadata):

        num_data, output_dim = Y.shape
        assert output_dim == 1, "This EP methods only works for 1D outputs"

        # Makes computing the sign quicker if we work with numpy arrays rather
        # than ObsArrays
        Y = Y.values.copy()

        #Initial values - Marginal moments, cavity params, gaussian approximation params and posterior params
        marg_moments = marginalMoments(num_data)
        cav_params = cavityParams(num_data)
        ga_approx, post_params, LLT0, LLT = self._init_approximations(Kmm, Kmn, num_data)

        #Approximation
        stop = False
        iterations = 0
        while not stop and (iterations < self.max_iters):
            self._local_updates(num_data, LLT0, LLT, Kmn, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata)
            #(re) compute Sigma, Sigma_diag and mu using full Cholesky decompy
            post_params, LLT = posteriorParamsDTC._recompute(LLT0, Kmn, ga_approx)
            post_params.Sigma_diag = np.maximum(post_params.Sigma_diag, np.finfo(float).eps)

            #monitor convergence
            if iterations > 0:
                stop = self._stop_criteria(ga_approx)
            self.ga_approx_old = gaussianApproximation(ga_approx.v.copy(), ga_approx.tau.copy())
            iterations += 1

        log_Z_tilde = self._log_Z_tilde(marg_moments, ga_approx, cav_params)

        return post_params, ga_approx, log_Z_tilde

    def _log_Z_tilde(self, marg_moments, ga_approx, cav_params):
        mu_tilde = ga_approx.v/ga_approx.tau
        mu_cav = cav_params.v/cav_params.tau
        sigma2_sigma2tilde = 1./cav_params.tau + 1./ga_approx.tau

        return np.sum((np.log(marg_moments.Z_hat) + 0.5*np.log(2*np.pi) + 0.5*np.log(sigma2_sigma2tilde)
                         + 0.5*((mu_cav - mu_tilde)**2) / (sigma2_sigma2tilde)))

    def _init_approximations(self, Kmm, Kmn, num_data):
        #initial values - Gaussian factors
        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        LLT0 = Kmm.copy()
        Lm = jitchol(LLT0) #K_m = L_m L_m^\top
        Vm,info = dtrtrs(Lm, Kmn,lower=1)
        # Lmi = dtrtri(Lm)
        # Kmmi = np.dot(Lmi.T,Lmi)
        # KmmiKmn = np.dot(Kmmi,Kmn)
        # Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        Qnn_diag = np.sum(Vm*Vm,-2) #diag(Knm Kmm^(-1) Kmn)
        #diag.add(LLT0, 1e-8)
        if self.ga_approx_old is None:
            #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
            LLT = LLT0.copy() #Sigma = K.copy()
            mu = np.zeros(num_data)
            Sigma_diag = Qnn_diag.copy() + 1e-8
            v_tilde, tau_tilde = np.zeros((2, num_data))
            ga_approx = gaussianApproximation(v_tilde, tau_tilde)
            post_params = posteriorParamsDTC(mu, Sigma_diag)

        else:
            assert self.ga_approx_old.v.size == num_data, "data size mis-match: did you change the data? try resetting!"
            ga_approx = gaussianApproximation(self.ga_approx_old.v, self.ga_approx_old.tau)
            post_params, LLT = posteriorParamsDTC._recompute(LLT0, Kmn, ga_approx)
            post_params.Sigma_diag += 1e-8

            # TODO: Check the log-marginal under both conditions and choose the best one

        return (ga_approx, post_params, LLT0, LLT)

    def _local_updates(self, num_data, LLT0, LLT, Kmn, cav_params, post_params, marg_moments, ga_approx, likelihood, Y, Y_metadata, update_order=None):
        if update_order is None:
            update_order = np.random.permutation(num_data)
        for i in update_order:

            #Cavity distribution parameters
            cav_params._update_i(self.eta, ga_approx, post_params, i)


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
            delta_tau, delta_v = ga_approx._update_i(self.eta, self.delta, post_params, marg_moments, i)

            #Posterior distribution parameters update
            if self.parallel_updates == False:
                post_params._update_rank1(LLT, Kmn, delta_v, delta_tau, i)


    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(EPDTC, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.expectation_propagation.EPDTC"
        if self.ga_approx_old is not  None:
            input_dict["ga_approx_old"] = self.ga_approx_old.to_dict()
        if self._ep_approximation is not  None:
            input_dict["_ep_approximation"] = {}
            input_dict["_ep_approximation"]["post_params"] = self._ep_approximation[0].to_dict()
            input_dict["_ep_approximation"]["ga_approx"] = self._ep_approximation[1].to_dict()
            input_dict["_ep_approximation"]["log_Z_tilde"] = self._ep_approximation[2]

        return input_dict

    @staticmethod
    def _build_from_input_dict(inference_class, input_dict):
        ga_approx_old = input_dict.pop('ga_approx_old', None)
        if ga_approx_old is not None:
            ga_approx_old = gaussianApproximation.from_dict(ga_approx_old)
        _ep_approximation_dict = input_dict.pop('_ep_approximation', None)
        _ep_approximation = []
        if _ep_approximation is not None:
            _ep_approximation.append(posteriorParamsDTC.from_dict(_ep_approximation_dict["post_params"]))
            _ep_approximation.append(gaussianApproximation.from_dict(_ep_approximation_dict["ga_approx"]))
            _ep_approximation.append(_ep_approximation_dict["log_Z_tilde"])
        ee = EPDTC(**input_dict)
        ee.ga_approx_old = ga_approx_old
        ee._ep_approximation = _ep_approximation
        return ee
