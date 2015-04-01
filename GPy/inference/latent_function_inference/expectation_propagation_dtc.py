# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ...util import diag
from ...util.linalg import mdot, jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri, dpotri, dpotrs, symmetrify, DSYR
from ...core.parameterization.variational import VariationalPosterior
from . import LatentFunctionInference
from .posterior import Posterior
log_2_pi = np.log(2*np.pi)

class EPDTC(LatentFunctionInference):
    const_jitter = 1e-6
    def __init__(self, epsilon=1e-6, eta=1., delta=1., limit=1):
        from ...util.caching import Cacher
        self.limit = limit
        self.get_trYYT = Cacher(self._get_trYYT, limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, limit)

        self.epsilon, self.eta, self.delta = epsilon, eta, delta
        self.reset()

    def set_limit(self, limit):
        self.get_trYYT.limit = limit
        self.get_YYTfactor.limit = limit

    def on_optimization_start(self):
        self._ep_approximation = None

    def on_optimization_end(self):
        # TODO: update approximation in the end as well? Maybe even with a switch?
        pass

    def _get_trYYT(self, Y):
        return np.sum(np.square(Y))

    def __getstate__(self):
        # has to be overridden, as Cacher objects cannot be pickled.
        return self.limit

    def __setstate__(self, state):
        # has to be overridden, as Cacher objects cannot be pickled.
        self.limit = state
        from ...util.caching import Cacher
        self.get_trYYT = Cacher(self._get_trYYT, self.limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, self.limit)

    def _get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT.

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>=D):
            return Y
        else:
            return jitchol(tdot(Y))

    def get_VVTfactor(self, Y, prec):
        return Y * prec # TODO chache this, and make it effective

    def reset(self):
        self.old_mutilde, self.old_vtilde = None, None
        self._ep_approximation = None

    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
        assert mean_function is None, "inference with a mean function not implemented"
        num_data, output_dim = Y.shape
        assert output_dim ==1, "ep in 1D only (for now!)"

        Kmm = kern.K(Z)
        Kmn = kern.K(Z,X)

        if self._ep_approximation is None:
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation = self.expectation_propagation(Kmm, Kmn, Y, likelihood, Y_metadata)
        else:
            mu, Sigma, mu_tilde, tau_tilde, Z_hat = self._ep_approximation


        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
            psi0 = kern.psi0(Z, X)
            psi1 = Kmn.T#kern.psi1(Z, X)
            psi2 = kern.psi2(Z, X)
        else:
            uncertain_inputs = False
            psi0 = kern.Kdiag(X)
            psi1 = Kmn.T#kern.K(X, Z)
            psi2 = None

        #see whether we're using variational uncertain inputs

        _, output_dim = Y.shape

        #see whether we've got a different noise variance for each datum
        #beta = 1./np.fmax(likelihood.gaussian_variance(Y_metadata), 1e-6)
        beta = tau_tilde
        VVT_factor = beta[:,None]*mu_tilde[:,None]
        trYYT = self.get_trYYT(mu_tilde[:,None])

        # do the inference:
        het_noise = beta.size > 1
        num_inducing = Z.shape[0]
        num_data = Y.shape[0]
        # kernel computations, using BGPLVM notation

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        Lm = jitchol(Kmm)

        # The rather complex computations of A
        if uncertain_inputs:
            if het_noise:
                psi2_beta = psi2 * (beta.flatten().reshape(num_data, 1, 1)).sum(0)
            else:
                psi2_beta = psi2.sum(0) * beta
            LmInv = dtrtri(Lm)
            A = LmInv.dot(psi2_beta.dot(LmInv.T))
        else:
            if het_noise:
                tmp = psi1 * (np.sqrt(beta.reshape(num_data, 1)))
            else:
                tmp = psi1 * (np.sqrt(beta))
            tmp, _ = dtrtrs(Lm, tmp.T, lower=1)
            A = tdot(tmp) #print A.sum()

        # factor B
        B = np.eye(num_inducing) + A
        LB = jitchol(B)
        psi1Vf = np.dot(psi1.T, VVT_factor)
        # back substutue C into psi1Vf
        tmp, _ = dtrtrs(Lm, psi1Vf, lower=1, trans=0)
        _LBi_Lmi_psi1Vf, _ = dtrtrs(LB, tmp, lower=1, trans=0)
        tmp, _ = dtrtrs(LB, _LBi_Lmi_psi1Vf, lower=1, trans=1)
        Cpsi1Vf, _ = dtrtrs(Lm, tmp, lower=1, trans=1)

        # data fit and derivative of L w.r.t. Kmm
        delit = tdot(_LBi_Lmi_psi1Vf)
        data_fit = np.trace(delit)
        DBi_plus_BiPBi = backsub_both_sides(LB, output_dim * np.eye(num_inducing) + delit)
        delit = -0.5 * DBi_plus_BiPBi
        delit += -0.5 * B * output_dim
        delit += output_dim * np.eye(num_inducing)
        # Compute dL_dKmm
        dL_dKmm = backsub_both_sides(Lm, delit)

        # derivatives of L w.r.t. psi
        dL_dpsi0, dL_dpsi1, dL_dpsi2 = _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm,
            VVT_factor, Cpsi1Vf, DBi_plus_BiPBi,
            psi1, het_noise, uncertain_inputs)

        # log marginal likelihood
        log_marginal = _compute_log_marginal_likelihood(likelihood, num_data, output_dim, beta, het_noise,
            psi0, A, LB, trYYT, data_fit, VVT_factor)

        #put the gradients in the right places
        dL_dR = _compute_dL_dR(likelihood,
            het_noise, uncertain_inputs, LB,
            _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A,
            psi0, psi1, beta,
            data_fit, num_data, output_dim, trYYT, mu_tilde[:,None])

        dL_dthetaL = 0#likelihood.exact_inference_gradients(dL_dR,Y_metadata)

        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dpsi0':dL_dpsi0,
                         'dL_dpsi1':dL_dpsi1,
                         'dL_dpsi2':dL_dpsi2,
                         'dL_dthetaL':dL_dthetaL}
        else:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dKdiag':dL_dpsi0,
                         'dL_dKnm':dL_dpsi1,
                         'dL_dthetaL':dL_dthetaL}

        #get sufficient things for posterior prediction
        #TODO: do we really want to do this in  the loop?
        if VVT_factor.shape[1] == Y.shape[1]:
            woodbury_vector = Cpsi1Vf # == Cpsi1V
        else:
            print('foobar')
            psi1V = np.dot(mu_tilde[:,None].T*beta, psi1).T
            tmp, _ = dtrtrs(Lm, psi1V, lower=1, trans=0)
            tmp, _ = dpotrs(LB, tmp, lower=1)
            woodbury_vector, _ = dtrtrs(Lm, tmp, lower=1, trans=1)
        Bi, _ = dpotri(LB, lower=1)
        symmetrify(Bi)
        Bi = -dpotri(LB, lower=1)[0]
        diag.add(Bi, 1)

        woodbury_inv = backsub_both_sides(Lm, Bi)

        #construct a posterior object
        post = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector, K=Kmm, mean=None, cov=None, K_chol=Lm)
        return post, log_marginal, grad_dict





    def expectation_propagation(self, Kmm, Kmn, Y, likelihood, Y_metadata):

        num_data, data_dim = Y.shape
        assert data_dim == 1, "This EP methods only works for 1D outputs"

        KmnKnm = np.dot(Kmn,Kmn.T)
        Lm = jitchol(Kmm)
        Lmi = dtrtrs(Lm,np.eye(Lm.shape[0]))[0] #chol_inv(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        KmmiKmn = np.dot(Kmmi,Kmn)
        Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        LLT0 = Kmm.copy()

        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(num_data)
        LLT = Kmm.copy() #Sigma = K.copy()
        Sigma_diag = Qnn_diag.copy()

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
                L = jitchol(LLT)

                V,info = dtrtrs(L,Kmn,lower=1)
                Sigma_diag = np.sum(V*V,-2)
                si = np.sum(V.T*V[:,i],-1)
                mu += (delta_v-delta_tau*mu[i])*si
                #mu = np.dot(Sigma, v_tilde)

            #(re) compute Sigma and mu using full Cholesky decompy
            LLT = LLT0 + np.dot(Kmn*tau_tilde[None,:],Kmn.T)
            L = jitchol(LLT)
            V,info = dtrtrs(L,Kmn,lower=1)
            V2,info = dtrtrs(L.T,V,lower=0)
            #Sigma_diag = np.sum(V*V,-2)
            #Knmv_tilde = np.dot(Kmn,v_tilde)
            #mu = np.dot(V2.T,Knmv_tilde)
            Sigma = np.dot(V2.T,V2)
            mu = np.dot(Sigma,v_tilde)

            #monitor convergence
            if iterations>0:
                tau_diff = np.mean(np.square(tau_tilde-tau_tilde_old))
                v_diff = np.mean(np.square(v_tilde-v_tilde_old))
            tau_tilde_old = tau_tilde.copy()
            v_tilde_old = v_tilde.copy()

            tau_diff = 0
            v_diff = 0
            iterations += 1

        mu_tilde = v_tilde/tau_tilde
        return mu, Sigma, mu_tilde, tau_tilde, Z_hat

def _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm, VVT_factor, Cpsi1Vf, DBi_plus_BiPBi, psi1, het_noise, uncertain_inputs):
    dL_dpsi0 = -0.5 * output_dim * (beta[:,None] * np.ones([num_data, 1])).flatten()
    dL_dpsi1 = np.dot(VVT_factor, Cpsi1Vf.T)
    dL_dpsi2_beta = 0.5 * backsub_both_sides(Lm, output_dim * np.eye(num_inducing) - DBi_plus_BiPBi)
    if het_noise:
        if uncertain_inputs:
            dL_dpsi2 = beta[:, None, None] * dL_dpsi2_beta[None, :, :]
        else:
            dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, (psi1 * beta.reshape(num_data, 1)).T).T
            dL_dpsi2 = None
    else:
        dL_dpsi2 = beta * dL_dpsi2_beta
        if uncertain_inputs:
            # repeat for each of the N psi_2 matrices
            dL_dpsi2 = np.repeat(dL_dpsi2[None, :, :], num_data, axis=0)
        else:
            # subsume back into psi1 (==Kmn)
            dL_dpsi1 += 2.*np.dot(psi1, dL_dpsi2)
            dL_dpsi2 = None

    return dL_dpsi0, dL_dpsi1, dL_dpsi2


def _compute_dL_dR(likelihood, het_noise, uncertain_inputs, LB, _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A, psi0, psi1, beta, data_fit, num_data, output_dim, trYYT, Y):
    # the partial derivative vector for the likelihood
    if likelihood.size == 0:
        # save computation here.
        dL_dR = None
    elif het_noise:
        if uncertain_inputs:
            raise NotImplementedError("heteroscedatic derivates with uncertain inputs not implemented")
        else:
            #from ...util.linalg import chol_inv
            #LBi = chol_inv(LB)
            LBi, _ = dtrtrs(LB,np.eye(LB.shape[0]))

            Lmi_psi1, nil = dtrtrs(Lm, psi1.T, lower=1, trans=0)
            _LBi_Lmi_psi1, _ = dtrtrs(LB, Lmi_psi1, lower=1, trans=0)

            dL_dR = -0.5 * beta + 0.5 * (beta*Y)**2
            dL_dR += 0.5 * output_dim * (psi0 - np.sum(Lmi_psi1**2,0))[:,None] * beta**2

            dL_dR += 0.5*np.sum(mdot(LBi.T,LBi,Lmi_psi1)*Lmi_psi1,0)[:,None]*beta**2

            dL_dR += -np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T * Y * beta**2
            dL_dR += 0.5*np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T**2 * beta**2
    else:
        # likelihood is not heteroscedatic
        dL_dR = -0.5 * num_data * output_dim * beta + 0.5 * trYYT * beta ** 2
        dL_dR += 0.5 * output_dim * (psi0.sum() * beta ** 2 - np.trace(A) * beta)
        dL_dR += beta * (0.5 * np.sum(A * DBi_plus_BiPBi) - data_fit)
    return dL_dR

def _compute_log_marginal_likelihood(likelihood, num_data, output_dim, beta, het_noise, psi0, A, LB, trYYT, data_fit,Y):
    #compute log marginal likelihood
    if het_noise:
        lik_1 = -0.5 * num_data * output_dim * np.log(2. * np.pi) + 0.5 * np.sum(np.log(beta)) - 0.5 * np.sum(beta * np.square(Y).sum(axis=-1))
        lik_2 = -0.5 * output_dim * (np.sum(beta.flatten() * psi0) - np.trace(A))
    else:
        lik_1 = -0.5 * num_data * output_dim * (np.log(2. * np.pi) - np.log(beta)) - 0.5 * beta * trYYT
        lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
    lik_3 = -output_dim * (np.sum(np.log(np.diag(LB))))
    lik_4 = 0.5 * data_fit
    log_marginal = lik_1 + lik_2 + lik_3 + lik_4
    return log_marginal
