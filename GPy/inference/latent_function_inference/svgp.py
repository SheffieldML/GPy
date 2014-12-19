from . import LatentFunctionInference
from ...util import linalg
from ...util import choleskies
import numpy as np
from posterior import Posterior

class SVGP(LatentFunctionInference):
    def likelihood_quadrature(self, Y, m, v):
        Ysign = np.where(Y==1,1,-1).flatten()
        from scipy import stats
        self.gh_x, self.gh_w = np.polynomial.hermite.hermgauss(20)

        #assume probit for now.
        X = self.gh_x[None,:]*np.sqrt(2.*v[:,None]) + (m*Ysign)[:,None]
        p = stats.norm.cdf(X)
        p = np.clip(p, 1e-9, 1.-1e-9) # for numerical stability
        N = stats.norm.pdf(X)
        F = np.log(p).dot(self.gh_w)
        NoverP = N/p
        dF_dm = (NoverP*Ysign[:,None]).dot(self.gh_w)
        dF_dv = -0.5*(NoverP**2 + NoverP*X).dot(self.gh_w)
        return F, dF_dm, dF_dv


    def inference(self, q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, Y_metadata=None):
        assert Y.shape[1]==1, "multi outputs not implemented"


        num_inducing = Z.shape[0]
        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol[:,None]).squeeze()
        S = L.dot(L.T)
        Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        logdetS = 2.*np.sum(np.log(np.abs(np.diag(L))))

        if np.any(np.isinf(Si)):
            print "warning:Cholesky representation unstable"
            S = S + np.eye(S.shape[0])*1e-5*np.max(np.max(S))
            Si, Lnew, _,_ = linalg.pdinv(S)

        #compute kernel related stuff
        Kmm = kern.K(Z)
        Knm = kern.K(X, Z)
        Knn_diag = kern.Kdiag(X)
        Kmmi, Lm, Lmi, logdetKmm = linalg.pdinv(Kmm)

        #compute the marginal means and variances of q(f)
        A = np.dot(Knm, Kmmi)
        mu = np.dot(A, q_u_mean)
        v = Knn_diag - np.sum(A*Knm,1) + np.sum(A*A.dot(S),1)

        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        KL = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi*S) + 0.5*q_u_mean.dot(Kmmim)
        dKL_dm = Kmmim
        dKL_dS = 0.5*(Kmmi - Si)
        dKL_dKmm = 0.5*Kmmi - 0.5*Kmmi.dot(S).dot(Kmmi) - 0.5*Kmmim[:,None]*Kmmim[None,:]

        #quadrature for the likelihood
        #F, dF_dmu, dF_dv = likelihood.variational_expectations(Y, mu, v)
        F, dF_dmu, dF_dv = self.likelihood_quadrature(Y, mu, v)


        #rescale the F term if working on a batch
        #F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale

        #derivatives of quadratured likelihood
        Adv = A.T*dF_dv # As if dF_Dv is diagonal
        Admu = A.T.dot(dF_dmu)
        AdvA = np.dot(Adv,A)
        tmp = AdvA.dot(S).dot(Kmmi)
        dF_dKmm = -Admu[:,None].dot(Kmmim[None,:]) + AdvA - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        dF_dKmn = 2.*(Kmmi.dot(S) - np.eye(num_inducing)).dot(Adv) + Kmmim[:,None]*dF_dmu[None,:]
        dF_dm = Admu
        dF_dS = AdvA

        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn

        dL_dchol = 2.*np.dot(dL_dS, L)
        dL_dchol = choleskies.triang_to_flat(dL_dchol[:,:,None]).squeeze()

        return Posterior(mean=q_u_mean, cov=S, K=Kmm), log_marginal, {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv, 'dL_dm':dL_dm, 'dL_dchol':dL_dchol}



