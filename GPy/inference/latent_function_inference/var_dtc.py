# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import Posterior
from ...util.linalg import mdot, jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri, dpotri, dpotrs, symmetrify
from ...util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)
import logging, itertools
logger = logging.getLogger('vardtc')

class VarDTC(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = 1e-8
    def __init__(self, limit=1):
        from paramz.caching import Cacher
        self.limit = limit
        self.get_trYYT = Cacher(self._get_trYYT, limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, limit)

    def set_limit(self, limit):
        self.get_trYYT.limit = limit
        self.get_YYTfactor.limit = limit

    def _get_trYYT(self, Y):
        return np.einsum("ij,ij->", Y, Y)
        # faster than, but same as:
        # return np.sum(np.square(Y))

    def __getstate__(self):
        # has to be overridden, as Cacher objects cannot be pickled.
        return self.limit

    def __setstate__(self, state):
        # has to be overridden, as Cacher objects cannot be pickled.
        self.limit = state
        from paramz.caching import Cacher
        self.get_trYYT = Cacher(self._get_trYYT, self.limit)
        self.get_YYTfactor = Cacher(self._get_YYTfactor, self.limit)

    def _get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT.

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>=D):
            return Y.view(np.ndarray)
        else:
            return jitchol(tdot(Y))

    def get_VVTfactor(self, Y, prec):
        return Y * prec # TODO chache this, and make it effective

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None, mean_function=None, precision=None, Lm=None, dL_dKmm=None, psi0=None, psi1=None, psi2=None, Z_tilde=None):
        assert mean_function is None, "inference with a mean function not implemented"

        num_data, output_dim = Y.shape
        num_inducing = Z.shape[0]

        uncertain_inputs = isinstance(X, VariationalPosterior)

        if precision is None:
            #assume Gaussian likelihood
            precision = 1./np.fmax(likelihood.gaussian_variance(Y_metadata), self.const_jitter)

        if precision.ndim == 1:
            precision = precision[:, None]
        het_noise = precision.size > 1

        VVT_factor = precision*Y
        #VVT_factor = precision*Y
        trYYT = self.get_trYYT(Y)

        # kernel computations, using BGPLVM notation
        if Lm is None:
            Kmm = kern.K(Z).copy()
            diag.add(Kmm, self.const_jitter)
            Lm = jitchol(Kmm)
        else:
            Kmm = tdot(Lm)
            symmetrify(Kmm)


        # The rather complex computations of A, and the psi stats
        if uncertain_inputs:
            if psi0 is None:
                psi0 = kern.psi0(Z, X)
            if psi1 is None:
                psi1 = kern.psi1(Z, X)
            if het_noise:
                if psi2 is None:
                    psi2_beta = (kern.psi2n(Z, X) * precision[:, :, None]).sum(0)
                else:
                    psi2_beta = (psi2 * precision[:, :, None]).sum(0)
            else:
                if psi2 is None:
                    psi2_beta = kern.psi2(Z,X) * precision
                elif psi2.ndim == 3:
                    psi2_beta = psi2.sum(0) * precision
                else:
                    psi2_beta = psi2 * precision
            LmInv = dtrtri(Lm)
            A = LmInv.dot(psi2_beta.dot(LmInv.T))
        else:
            if psi0 is None:
                psi0 = kern.Kdiag(X)
            if psi1 is None:
                psi1 = kern.K(X, Z)
            if het_noise:
                tmp = psi1 * (np.sqrt(precision))
            else:
                tmp = psi1 * (np.sqrt(precision))
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
        if dL_dKmm is None:
            delit = -0.5 * DBi_plus_BiPBi
            delit += -0.5 * B * output_dim
            delit += output_dim * np.eye(num_inducing)
            # Compute dL_dKmm
            dL_dKmm = backsub_both_sides(Lm, delit)

        # derivatives of L w.r.t. psi
        dL_dpsi0, dL_dpsi1, dL_dpsi2 = _compute_dL_dpsi(num_inducing, num_data, output_dim, precision, Lm,
            VVT_factor, Cpsi1Vf, DBi_plus_BiPBi,
            psi1, het_noise, uncertain_inputs)

        # log marginal likelihood
        log_marginal = _compute_log_marginal_likelihood(likelihood, num_data, output_dim, precision, het_noise,
            psi0, A, LB, trYYT, data_fit, Y)

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        #noise derivatives
        dL_dR = _compute_dL_dR(likelihood,
            het_noise, uncertain_inputs, LB,
            _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A,
            psi0, psi1, precision,
            data_fit, num_data, output_dim, trYYT, Y, VVT_factor)

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR,Y_metadata)

        #put the gradients in the right places
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
            import ipdb; ipdb.set_trace()
            psi1V = np.dot(Y.T*precision, psi1).T
            tmp, _ = dtrtrs(Lm, psi1V, lower=1, trans=0)
            tmp, _ = dpotrs(LB, tmp, lower=1)
            woodbury_vector, _ = dtrtrs(Lm, tmp, lower=1, trans=1)
        #Bi, _ = dpotri(LB, lower=1)
        #symmetrify(Bi)
        Bi = -dpotri(LB, lower=1)[0]
        diag.add(Bi, 1)

        woodbury_inv = backsub_both_sides(Lm, Bi)

        #construct a posterior object
        post = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector, K=Kmm, mean=None, cov=None, K_chol=Lm)
        return post, log_marginal, grad_dict

def _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm, VVT_factor, Cpsi1Vf, DBi_plus_BiPBi, psi1, het_noise, uncertain_inputs):
    dL_dpsi0 = -0.5 * output_dim * (beta* np.ones([num_data, 1])).flatten()
    dL_dpsi1 = np.dot(VVT_factor, Cpsi1Vf.T)
    dL_dpsi2_beta = 0.5 * backsub_both_sides(Lm, output_dim * np.eye(num_inducing) - DBi_plus_BiPBi)
    if het_noise:
        if uncertain_inputs:
            dL_dpsi2 = beta[:, None] * dL_dpsi2_beta[None, :, :]
        else:
            dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, (psi1 * beta).T).T
            dL_dpsi2 = None
    else:
        dL_dpsi2 = beta * dL_dpsi2_beta
        if not uncertain_inputs:
            # subsume back into psi1 (==Kmn)
            dL_dpsi1 += 2.*np.dot(psi1, dL_dpsi2)
            dL_dpsi2 = None
    return dL_dpsi0, dL_dpsi1, dL_dpsi2


def _compute_dL_dR(likelihood, het_noise, uncertain_inputs, LB, _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A, psi0, psi1, beta, data_fit, num_data, output_dim, trYYT, Y, VVT_factr=None):
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
            dL_dR = -0.5 * beta + 0.5 * VVT_factr**2
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

def _compute_log_marginal_likelihood(likelihood, num_data, output_dim, beta, het_noise, psi0, A, LB, trYYT, data_fit, Y):
    #compute log marginal likelihood
    if het_noise:
        lik_1 = -0.5 * num_data * output_dim * np.log(2. * np.pi) + 0.5 * output_dim * np.sum(np.log(beta)) - 0.5 * np.sum(beta.ravel() * np.square(Y).sum(axis=-1))
        lik_2 = -0.5 * output_dim * (np.sum(beta.flatten() * psi0) - np.trace(A))
    else:
        lik_1 = -0.5 * num_data * output_dim * (np.log(2. * np.pi) - np.log(beta)) - 0.5 * beta * trYYT
        lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
    lik_3 = -output_dim * (np.sum(np.log(np.diag(LB))))
    lik_4 = 0.5 * data_fit
    log_marginal = lik_1 + lik_2 + lik_3 + lik_4
    return log_marginal
