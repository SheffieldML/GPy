from . import LatentFunctionInference
from ...util import linalg
from ...util import choleskies
import numpy as np
from posterior import Posterior

class SVGP(LatentFunctionInference):
    def inference(self, q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, Y_metadata=None):
        assert Y.shape[1]==1, "multi outputs not implemented"

        num_inducing = Z.shape[0]
        num_data, num_outputs = Y.shape
        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)
        S = np.einsum('ijk,ljk->ilk', L, L) #L.dot(L.T)
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(L)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[:,:,i])))) for i in range(L.shape[-1])])

        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")
            #S = S + np.eye(S.shape[0])*1e-5*np.max(np.max(S))
            #Si, Lnew, _,_ = linalg.pdinv(S)

        #compute kernel related stuff
        Kmm = kern.K(Z)
        Knm = kern.K(X, Z)
        Knn_diag = kern.Kdiag(X)
        Kmmi, Lm, Lmi, logdetKmm = linalg.pdinv(Kmm)

        #compute the marginal means and variances of q(f)
        A = np.dot(Knm, Kmmi)
        mu = np.dot(A, q_u_mean)
        #v = Knn_diag - np.sum(A*Knm,1) + np.sum(A*A.dot(S),1)
        v = Knn_diag[:,None] - np.sum(A*Knm,1)[:,None] + np.sum(A[:,:,None] * np.einsum('ij,jkl->ikl', A, S),1)

        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        #KL = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi*S) + 0.5*q_u_mean.dot(Kmmim)
        KLs = -0.5*logdetS -0.5*self.num_inducing + 0.5*logdetKmm + 0.5*np.einsum('ij,ijk->k', Kmmi, S) + 0.5*np.sum(self.q_u_mean*Kmmim,0)
        KL = KLs.sum()
        dKL_dm = Kmmim
        #dKL_dS = 0.5*(Kmmi - Si)
        dKL_dS = 0.5*(Kmmi[:,:,None] - Si)
        #dKL_dKmm = 0.5*Kmmi - 0.5*Kmmi.dot(S).dot(Kmmi) - 0.5*Kmmim[:,None]*Kmmim[None,:]
        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(-1)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)

        #if self.KL_scale:
            #scale = 1./np.float64(self.mpi_comm.size)
            #KL, dKL_dKmm, dKL_dS, dKL_dm = scale*KL, scale*dKL_dKmm, scale*dKL_dS, scale*dKL_dm

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v)

        #rescale the F term if working on a batch
        #F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale

        #derivatives of expected likelihood
        Adv = A.T[:,:,None]*dF_dv[None,:,:] # As if dF_Dv is diagonal
        Admu = A.T.dot(dF_dmu)
        #AdvA = np.einsum('ijk,jl->ilk', Adv, A) 
        #AdvA = np.dot(A.T, Adv).swapaxes(0,1)
        AdvA = np.dstack([np.dot(A.T, Adv[:,:,i].T) for i in range(self.num_classes)])
        tmp = np.einsum('ijk,jlk->il', AdvA, S).dot(Kmmi)
        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(-1) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        tmp = 2.*(np.einsum('ij,jlk->ilk', Kmmi,S) - np.eye(self.num_inducing)[:,:,None])
        dF_dKmn = np.einsum('ijk,jlk->il', tmp, Adv) + Kmmim.dot(dF_dmu.T)
        dF_dm = Admu
        dF_dS = AdvA


        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn

        dL_dchol = 2.*np.dot(dL_dS, L)
        dL_dchol = choleskies.triang_to_flat(dL_dchol)

        return Posterior(mean=q_u_mean, cov=S, K=Kmm), log_marginal, {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv, 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}



