# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from .../util.linalg import pdinv, dpotrs, tdot
log_2_pi = np.log(2*np.pi)

class DTCVar(object):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        self._YYTfactor_cache = caching.cache()
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
            #if Y in self.cache, return self.Cache[Y], else stor Y in cache and return L.
            raise NotImplementedError, 'TODO' #TODO

    def inference(self, Kmm, Kmn, Knn_diag, likelihood, Y):

        num_inducing, num_data = Kmn.shape
        const_jitter = np.eye(num_inducing) * self.const_jitter

        #factor Kmm # TODO: cache?
        _Lm = jitchol(Kmm + _const_jitter)

        # The rather complex computations of A
        if has_uncertain_inputs:
            if likelihood.is_heteroscedastic:
                psi2_beta = (psi2 * (likelihood.precision.flatten().reshape(num_data, 1, 1))).sum(0)
            else:
                psi2_beta = psi2.sum(0) * likelihood.precision
            evals, evecs = linalg.eigh(psi2_beta)
            clipped_evals = np.clip(evals, 0., 1e6) # TODO: make clipping configurable
            if not np.array_equal(evals, clipped_evals):
                pass # print evals
            tmp = evecs * np.sqrt(clipped_evals)
            tmp = tmp.T
        else:
            if likelihood.is_heteroscedastic:
                tmp = psi1 * (np.sqrt(likelihood.precision.flatten().reshape(num_data, 1)))
            else:
                tmp = psi1 * (np.sqrt(likelihood.precision))
        tmp, _ = dtrtrs(_Lm, np.asfortranarray(tmp.T), lower=1)
        A = tdot(tmp)

        # factor B
        B = np.eye(num_inducing) + A
        LB = jitchol(B)

        # VVT_factor is a matrix such that tdot(VVT_factor) = VVT...this is for efficiency!
        psi1Vf = np.dot(psi1.T, likelihood.VVT_factor)

        # back substutue C into psi1Vf
        tmp, info1 = dtrtrs(_Lm, np.asfortranarray(psi1Vf), lower=1, trans=0)
        _LBi_Lmi_psi1Vf, _ = dtrtrs(LB, np.asfortranarray(tmp), lower=1, trans=0)
        # tmp, info2 = dpotrs(LB, tmp, lower=1)
        tmp, info2 = dtrtrs(LB, _LBi_Lmi_psi1Vf, lower=1, trans=1)
        Cpsi1Vf, info3 = dtrtrs(_Lm, tmp, lower=1, trans=1)

        # Compute dL_dKmm
        tmp = tdot(_LBi_Lmi_psi1Vf)
        data_fit = np.trace(tmp)
        DBi_plus_BiPBi = backsub_both_sides(LB, output_dim * np.eye(num_inducing) + tmp)
        tmp = -0.5 * DBi_plus_BiPBi
        tmp += -0.5 * B * output_dim
        tmp += output_dim * np.eye(num_inducing)
        dL_dKmm = backsub_both_sides(_Lm, tmp)

        # Compute dL_dpsi # FIXME: this is untested for the heterscedastic + uncertain inputs case
        dL_dpsi0 = -0.5 * output_dim * (likelihood.precision * np.ones([num_data, 1])).flatten()
        dL_dpsi1 = np.dot(likelihood.VVT_factor, Cpsi1Vf.T)
        dL_dpsi2_beta = 0.5 * backsub_both_sides(_Lm, output_dim * np.eye(num_inducing) - DBi_plus_BiPBi)

        if likelihood.is_heteroscedastic:

            if has_uncertain_inputs:
                dL_dpsi2 = likelihood.precision.flatten()[:, None, None] * dL_dpsi2_beta[None, :, :]
            else:
                dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, (psi1 * likelihood.precision.reshape(num_data, 1)).T).T
                dL_dpsi2 = None
        else:
            dL_dpsi2 = likelihood.precision * dL_dpsi2_beta
            if has_uncertain_inputs:
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
        elif likelihood.is_heteroscedastic:

            if has_uncertain_inputs:
                raise NotImplementedError, "heteroscedatic derivates with uncertain inputs not implemented"

            else:

                LBi = chol_inv(LB)
                Lmi_psi1, nil = dtrtrs(_Lm, np.asfortranarray(psi1.T), lower=1, trans=0)
                _LBi_Lmi_psi1, _ = dtrtrs(LB, np.asfortranarray(Lmi_psi1), lower=1, trans=0)


                partial_for_likelihood = -0.5 * likelihood.precision + 0.5 * likelihood.V**2
                partial_for_likelihood += 0.5 * output_dim * (psi0 - np.sum(Lmi_psi1**2,0))[:,None] * likelihood.precision**2

                partial_for_likelihood += 0.5*np.sum(mdot(LBi.T,LBi,Lmi_psi1)*Lmi_psi1,0)[:,None]*likelihood.precision**2

                partial_for_likelihood += -np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T * likelihood.Y * likelihood.precision**2
                partial_for_likelihood += 0.5*np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T**2 * likelihood.precision**2

        else:
            # likelihood is not heteroscedatic
            partial_for_likelihood = -0.5 * num_data * output_dim * likelihood.precision + 0.5 * likelihood.trYYT * likelihood.precision ** 2
            partial_for_likelihood += 0.5 * output_dim * (psi0.sum() * likelihood.precision ** 2 - np.trace(_A) * likelihood.precision)
            partial_for_likelihood += likelihood.precision * (0.5 * np.sum(_A * DBi_plus_BiPBi) - data_fit)

    #def log_likelihood(self):
        if likelihood.is_heteroscedastic:
            A = -0.5 * num_data * output_dim * np.log(2.*np.pi) + 0.5 * np.sum(np.log(likelihood.precision)) - 0.5 * np.sum(likelihood.V * likelihood.Y)
            B = -0.5 * output_dim * (np.sum(likelihood.precision.flatten() * psi0) - np.trace(_A))
        else:
            A = -0.5 * num_data * output_dim * (np.log(2.*np.pi) - np.log(likelihood.precision)) - 0.5 * likelihood.precision * likelihood.trYYT
            B = -0.5 * output_dim * (np.sum(likelihood.precision * psi0) - np.trace(_A))
        C = -output_dim * (np.sum(np.log(np.diag(LB)))) # + 0.5 * num_inducing * np.log(sf2))
        D = 0.5 * data_fit
        return A + B + C + D + likelihood.Z

