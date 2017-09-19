


from GPy.util.linalg import jitchol, backsub_both_sides, tdot, dtrtrs, dtrtri,pdinv, dpotri
from GPy.util import diag
from GPy.core.parameterization.variational import VariationalPosterior
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior

log_2_pi = np.log(2*np.pi)

class VarDTC_MD(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian, but we want to do sparse inference.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    const_jitter = 1e-6

    def gatherPsiStat(self, kern, X, Z, Y, beta, uncertain_inputs):

        if uncertain_inputs:
            psi0 = kern.psi0(Z, X)
            psi1 = kern.psi1(Z, X)
            psi2 = kern.psi2n(Z, X)
        else:
            psi0 = kern.Kdiag(X)
            psi1 = kern.K(X, Z)
            psi2 = psi1[:,:,None]*psi1[:,None,:]

        return psi0, psi1, psi2

    def inference(self, kern, X, Z, likelihood, Y, indexD, output_dim, Y_metadata=None, Lm=None, dL_dKmm=None, Kuu_sigma=None):
        """
        The first phase of inference:
        Compute: log-likelihood, dL_dKmm

        Cached intermediate results: Kmm, KmmInv,
        """

        input_dim = Z.shape[0]

        uncertain_inputs = isinstance(X, VariationalPosterior)

        beta = 1./likelihood.variance
        if len(beta)==1:
            beta = np.zeros(output_dim)+beta

        beta_exp = np.zeros(indexD.shape[0])
        for d in range(output_dim):
            beta_exp[indexD==d] = beta[d]

        psi0, psi1, psi2 = self.gatherPsiStat(kern, X, Z, Y, beta, uncertain_inputs)

        psi2_sum = (beta_exp[:,None,None]*psi2).sum(0)/output_dim

        #======================================================================
        # Compute Common Components
        #======================================================================

        Kmm = kern.K(Z).copy()
        if Kuu_sigma is not None:
            diag.add(Kmm, Kuu_sigma)
        else:
            diag.add(Kmm, self.const_jitter)
        Lm = jitchol(Kmm)

        logL = 0.
        dL_dthetaL = np.zeros(output_dim)
        dL_dKmm = np.zeros_like(Kmm)
        dL_dpsi0 = np.zeros_like(psi0)
        dL_dpsi1 = np.zeros_like(psi1)
        dL_dpsi2 = np.zeros_like(psi2)
        wv = np.empty((Kmm.shape[0],output_dim))

        for d in range(output_dim):
            idx_d = indexD==d
            Y_d = Y[idx_d]
            N_d = Y_d.shape[0]
            beta_d = beta[d]

            psi2_d = psi2[idx_d].sum(0)*beta_d
            psi1Y = Y_d.T.dot(psi1[idx_d])*beta_d
            psi0_d = psi0[idx_d].sum()*beta_d
            YRY_d = np.square(Y_d).sum()*beta_d

            LmInvPsi2LmInvT = backsub_both_sides(Lm, psi2_d, 'right')

            Lambda = np.eye(Kmm.shape[0])+LmInvPsi2LmInvT
            LL = jitchol(Lambda)
            LmLL = Lm.dot(LL)

            b  = dtrtrs(LmLL, psi1Y.T)[0].T
            bbt = np.square(b).sum()
            v = dtrtrs(LmLL, b.T, trans=1)[0].T
            LLinvPsi1TYYTPsi1LLinvT = tdot(b.T)

            tmp = -backsub_both_sides(LL, LLinvPsi1TYYTPsi1LLinvT)
            dL_dpsi2R = backsub_both_sides(Lm, tmp+np.eye(input_dim))/2

            logL_R = -N_d*np.log(beta_d)
            logL += -((N_d*log_2_pi+logL_R+psi0_d-np.trace(LmInvPsi2LmInvT))+YRY_d- bbt)/2.

            dL_dKmm +=  dL_dpsi2R - backsub_both_sides(Lm, LmInvPsi2LmInvT)/2

            dL_dthetaL[d:d+1] = (YRY_d*beta_d + beta_d*psi0_d - N_d*beta_d)/2. - beta_d*(dL_dpsi2R*psi2_d).sum() - beta_d*np.trace(LLinvPsi1TYYTPsi1LLinvT)

            dL_dpsi0[idx_d] = -beta_d/2.
            dL_dpsi1[idx_d] = beta_d*np.dot(Y_d,v)
            dL_dpsi2[idx_d] = beta_d*dL_dpsi2R
            wv[:,d] = v

        LmInvPsi2LmInvT = backsub_both_sides(Lm, psi2_sum, 'right')

        Lambda = np.eye(Kmm.shape[0])+LmInvPsi2LmInvT
        LL = jitchol(Lambda)
        LmLL = Lm.dot(LL)
        logdet_L = 2.*np.sum(np.log(np.diag(LL)))
        dL_dpsi2R_common = dpotri(LmLL)[0]/-2.
        dL_dpsi2 += dL_dpsi2R_common[None,:,:]*beta_exp[:,None,None]

        for d in range(output_dim):
            dL_dthetaL[d] += (dL_dpsi2R_common*psi2[indexD==d].sum(0)).sum()*-beta[d]*beta[d]

        dL_dKmm += dL_dpsi2R_common*output_dim

        logL += -output_dim*logdet_L/2.

        #======================================================================
        # Compute dL_dKmm
        #======================================================================

        # dL_dKmm =  dL_dpsi2R - output_dim* backsub_both_sides(Lm, LmInvPsi2LmInvT)/2 #LmInv.T.dot(LmInvPsi2LmInvT).dot(LmInv)/2.

        #======================================================================
        # Compute the Posterior distribution of inducing points p(u|Y)
        #======================================================================

        LLInvLmT = dtrtrs(LL, Lm.T)[0]
        cov = tdot(LLInvLmT.T)

        wd_inv = backsub_both_sides(Lm, np.eye(input_dim)- backsub_both_sides(LL, np.identity(input_dim), transpose='left'), transpose='left')
        post = Posterior(woodbury_inv=wd_inv, woodbury_vector=wv, K=Kmm, mean=None, cov=cov, K_chol=Lm)

        #======================================================================
        # Compute dL_dthetaL for uncertian input and non-heter noise
        #======================================================================

        # for d in range(output_dim):
        #     dL_dthetaL[d:d+1] += - beta[d]*beta[d]*(dL_dpsi2R[None,:,:] * psi2[indexD==d]/output_dim).sum()
        # dL_dthetaL += - (dL_dpsi2R[None,:,:] * psi2_sum*D beta*(dL_dpsi2R*psi2).sum()

        #======================================================================
        # Compute dL_dpsi
        #======================================================================

        if not uncertain_inputs:
            dL_dpsi1 += (psi1[:,None,:]*dL_dpsi2).sum(2)*2.

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

        return post, logL, grad_dict
