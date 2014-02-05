# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from ...util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dpotri, symmetrify
import numpy as np
from GPy.util.linalg import dtrtri
log_2_pi = np.log(2*np.pi)

class VarDTC(object):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        #self._YYTfactor_cache = caching.cache()
        self.const_jitter = 1e-6

    def get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT. 

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>D):
            return Y
        else:
            #if Y in self.cache, return self.Cache[Y], else store Y in cache and return L.
            raise NotImplementedError, 'TODO' #TODO

    def get_VVTfactor(self, Y, prec):
        return Y * prec # TODO chache this, and make it effective

    def inference(self, kern, X, X_variance, Z, likelihood, Y):

        num_inducing, _ = Z.shape
        num_data, output_dim = Y.shape

        #see whether we're using variational uncertain inputs
        uncertain_inputs = not (X_variance is None)

        #see whether we've got a different noise variance for each datum
        beta = 1./np.squeeze(likelihood.variance)
        het_noise = False
        if beta.size <1:
            het_noise = True

        # kernel computations, using BGPLVM notation
        Kmm = kern.K(Z)
        if uncertain_inputs:
            psi0 = kern.psi0(Z, X, X_variance)
            psi1 = kern.psi1(Z, X, X_variance)
            psi2 = kern.psi2(Z, X, X_variance)
        else:
            psi0 = kern.Kdiag(X)
            psi1 = kern.K(X, Z)

        #factor Kmm # TODO: cache?
        Lm = jitchol(Kmm)

        # The rather complex computations of A
        if uncertain_inputs:
            if het_noise:
                psi2_beta = (psi2 * (beta.flatten().reshape(num_data, 1, 1))).sum(0)
            else:
                psi2_beta = psi2.sum(0) * beta
            if 0:
                evals, evecs = linalg.eigh(psi2_beta)
                clipped_evals = np.clip(evals, 0., 1e6) # TODO: make clipping configurable
                if not np.array_equal(evals, clipped_evals):
                    pass # print evals
                tmp = evecs * np.sqrt(clipped_evals)
                tmp = tmp.T
            # no backsubstitution because of bound explosion on tr(A) if not...
            LmInv, _ = dtrtri(Lm, lower=1)
            A = LmInv.T.dot(psi2_beta.dot(LmInv))
            print A.sum()
        else:
            if het_noise:
                tmp = psi1 * (np.sqrt(beta.reshape(num_data, 1)))
            else:
                tmp = psi1 * (np.sqrt(beta))
            tmp, _ = dtrtrs(Lm, np.asfortranarray(tmp.T), lower=1)
            A = tdot(tmp)

        # factor B
        B = np.eye(num_inducing) + A
        LB = jitchol(B)

        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        VVT_factor = self.get_VVTfactor(Y, beta)
        trYYT = np.sum(np.square(Y))
        psi1Vf = np.dot(psi1.T, VVT_factor)

        # back substutue C into psi1Vf
        tmp, info1 = dtrtrs(Lm, np.asfortranarray(psi1Vf), lower=1, trans=0)
        _LBi_Lmi_psi1Vf, _ = dtrtrs(LB, np.asfortranarray(tmp), lower=1, trans=0)
        tmp, info2 = dtrtrs(LB, _LBi_Lmi_psi1Vf, lower=1, trans=1)
        Cpsi1Vf, info3 = dtrtrs(Lm, tmp, lower=1, trans=1)


        # Compute dL_dKmm
        tmp = tdot(_LBi_Lmi_psi1Vf)
        data_fit = np.trace(tmp)
        DBi_plus_BiPBi = backsub_both_sides(LB, output_dim * np.eye(num_inducing) + tmp)
        tmp = -0.5 * DBi_plus_BiPBi
        tmp += -0.5 * B * output_dim
        tmp += output_dim * np.eye(num_inducing)
        dL_dKmm = backsub_both_sides(Lm, tmp)

        # Compute dL_dpsi 
        dL_dpsi0 = -0.5 * output_dim * (beta * np.ones([num_data, 1])).flatten()
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


        # the partial derivative vector for the likelihood
        if likelihood.size == 0:
            # save computation here.
            partial_for_likelihood = None
        elif het_noise:
            if uncertain_inputs:
                raise NotImplementedError, "heteroscedatic derivates with uncertain inputs not implemented"
            else:
                LBi = chol_inv(LB)
                Lmi_psi1, nil = dtrtrs(Lm, np.asfortranarray(psi1.T), lower=1, trans=0)
                _LBi_Lmi_psi1, _ = dtrtrs(LB, np.asfortranarray(Lmi_psi1), lower=1, trans=0)

                partial_for_likelihood = -0.5 * beta + 0.5 * likelihood.V**2
                partial_for_likelihood += 0.5 * output_dim * (psi0 - np.sum(Lmi_psi1**2,0))[:,None] * beta**2

                partial_for_likelihood += 0.5*np.sum(mdot(LBi.T,LBi,Lmi_psi1)*Lmi_psi1,0)[:,None]*beta**2

                partial_for_likelihood += -np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T * likelihood.Y * beta**2
                partial_for_likelihood += 0.5*np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T**2 * beta**2

        else:
            # likelihood is not heteroscedatic
            partial_for_likelihood = -0.5 * num_data * output_dim * beta + 0.5 * trYYT * beta ** 2
            partial_for_likelihood += 0.5 * output_dim * (psi0.sum() * beta ** 2 - np.trace(A) * beta)
            partial_for_likelihood += beta * (0.5 * np.sum(A * DBi_plus_BiPBi) - data_fit)

        #compute log marginal likelihood
        if het_noise:
            lik_1 = -0.5 * num_data * output_dim * np.log(2.*np.pi) + 0.5 * np.sum(np.log(beta)) - 0.5 * np.sum(likelihood.V * likelihood.Y)
            lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
        else:
            lik_1 = -0.5 * num_data * output_dim * (np.log(2.*np.pi) - np.log(beta)) - 0.5 * beta * trYYT
            lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
        lik_3 = -output_dim * (np.sum(np.log(np.diag(LB))))
        lik_4 = 0.5 * data_fit
        log_marginal = lik_1 + lik_2 + lik_3 + lik_4

        #put the gradients in the right places
        likelihood.update_gradients(partial_for_likelihood)

        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm, 'dL_dpsi0':dL_dpsi0, 'dL_dpsi1':dL_dpsi1, 'dL_dpsi2':dL_dpsi2}
            kern.update_gradients_variational(mu=X, S=X_variance, Z=Z, **grad_dict)
        else:
            grad_dict = {'dL_dKmm': dL_dKmm, 'dL_dKdiag':dL_dpsi0, 'dL_dKnm':dL_dpsi1}
            kern.update_gradients_sparse(X=X, Z=Z, **grad_dict)

        #get sufficient things for posterior prediction
        #TODO: do we really want to do this in  the loop?
        if VVT_factor.shape[1] == Y.shape[1]:
            woodbury_vector = Cpsi1Vf # == Cpsi1V
        else:
            psi1V = np.dot(Y.T*beta, psi1).T
            tmp, _ = dtrtrs(Lm, np.asfortranarray(psi1V), lower=1, trans=0)
            tmp, _ = dpotrs(LB, tmp, lower=1)
            woodbury_vector, _ = dtrtrs(Lm, tmp, lower=1, trans=1)
        Bi, _ = dpotri(LB, lower=0)
        symmetrify(Bi)
        woodbury_inv = backsub_both_sides(Lm, np.eye(num_inducing) - Bi)


        #construct a posterior object
        post = Posterior(woodbury_inv=woodbury_inv, woodbury_vector=woodbury_vector, K=Kmm, mean=None, cov=None, K_chol=Lm)

        return post, log_marginal, grad_dict


