from . import LatentFunctionInference
from ...util import linalg
from ...util import choleskies
import numpy as np
from .posterior import Posterior
from scipy.linalg.blas import dgemm, dsymm, dtrmm

class SVGP(LatentFunctionInference):

    def inference(self, q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, KL_scale=1.0, batch_scale=1.0):

        num_data, _ = Y.shape
        num_inducing, num_outputs = q_u_mean.shape

        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)


        S = np.empty((num_outputs, num_inducing, num_inducing))
        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(L)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])

        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")

        #compute mean function stuff
        if mean_function is not None:
            prior_mean_u = mean_function.f(Z)
            prior_mean_f = mean_function.f(X)
        else:
            prior_mean_u = np.zeros((num_inducing, num_outputs))
            prior_mean_f = np.zeros((num_data, num_outputs))

        #compute kernel related stuff
        Kmm = kern.K(Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        Lm = linalg.jitchol(Kmm)
        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
        Kmmi, _ = linalg.dpotri(Lm)

        #compute the marginal means and variances of q(f)
        A, _ = linalg.dpotrs(Lm, Kmn)
        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
        v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
            v[:,i] = np.sum(np.square(tmp),0)
        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]

        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KL = KLs.sum()
        #gradient of the KL term (assuming zero mean function)
        dKL_dm = Kmmim.copy()
        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)

        if mean_function is not None:
            #adjust KL term for mean function
            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
            KL += -np.sum(q_u_mean*Kmmi_mfZ)
            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)

            #adjust gradient for mean fucntion
            dKL_dm -= Kmmi_mfZ
            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)

            #compute gradients for mean_function
            dKL_dmfZ = Kmmi_mfZ - Kmmim

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)

        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #derivatives of expected likelihood, assuming zero mean function
        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
        Admu = A.dot(dF_dmu)
        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])

        dF_dKmn = Kmmim.dot(dF_dmu.T)
        for a,b in zip(tmp, Adv):
            dF_dKmn += np.dot(a.T, b)

        dF_dm = Admu
        dF_dS = AdvA

        #adjust gradient to account for mean function
        if mean_function is not None:
            dF_dmfX = dF_dmu.copy()
            dF_dmfZ = -Admu
            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
            dF_dKmm += Admu.dot(Kmmi_mfZ.T)


        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn

        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
        dL_dchol = choleskies.triang_to_flat(dL_dchol)

        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
        if mean_function is not None:
            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
            grad_dict['dL_dmfX'] = dF_dmfX
        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict
