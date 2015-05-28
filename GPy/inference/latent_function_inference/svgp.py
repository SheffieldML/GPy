from . import LatentFunctionInference
from ...util import linalg
from ...util import choleskies
import numpy as np
from .posterior import Posterior
from scipy.linalg.blas import dgemm, dsymm, dtrmm

class SVGP(LatentFunctionInference):

    def inference(self, q_v_mean, q_v_chol, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, KL_scale=1.0, batch_scale=1.0):

        if mean_function is not None:
            raise NotImplementedError

        num_data, _ = Y.shape
        num_inducing, num_outputs = q_v_mean.shape

        #expand cholesky representation
        Lv = choleskies.flat_to_triang(q_v_chol)

        #deal with posterior copvariance
        Sv = np.zeros((num_outputs, num_inducing, num_inducing))
        for i in range(num_outputs):
            Sv[i] = Lv[i].dot(Lv[i].T)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(Lv[i])))) for i in range(num_outputs)])
        traceS = np.array([np.sum(np.square(np.diag(Lv[i]))) for i in range(num_outputs)])


        #compute kernel related stuff
        Kmm = kern.K(Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        R = linalg.jitchol(Kmm)

        #compute the marginal means and variances of q(f)
        AT, _ = linalg.dtrtrs(R, Kmn)
        A = AT.T
        mu = np.dot(A, q_v_mean)
        var = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,Lv[i].T, A.T, lower=0, trans_a=0)
            var[:,i] = np.sum(np.square(tmp),0)
        var += (Knn_diag - np.sum(np.square(A),1))[:,None]

        #compute the KL term
        KL = -0.5*logdetS.sum() + 0.5*np.sum(np.square(q_v_mean)) + 0.5*traceS.sum() - 0.5*num_inducing*output_dim
        dL_dmv = -q_v_mean*1
        dL_dL = np.zeros_like(Lv)
        for i in range(num_outputs):
            Lii = np.diagonal(Lv[i])
            dL_dL[i] -= np.diag(Lii - 1./Lii)

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, var, Y_metadata=Y_metadata)

        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale

        #sum over the data for the gradients of the likelihood parameters
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #mv
        dL_dmv += A.T.dot(dF_dmu)

        # A
        dL_dA_via_v = np.zeros(A.shape)
        for i in range(num_outputs):
            dL_dA_via_v += -2*(np.eye(num_inducing) - Sv[i]).dot(A.T * dF_dv[:,i]).T

        #Kfu
        RiTm, _ = linalg.dtrtrs(R, q_v_mean, lower=1, trans=1)
        dL_dKmn, _ = linalg.dtrtrs(R, dL_dA_via_v.T, trans=1, lower=1)
        dL_dKmn += np.dot(RiTm, dF_dmu.T)

        #L
        for i in range(num_outputs):
            dL_dL[i] += 2.*np.dot(Lv[i].T, A.T).dot(A*dF_dv[:,i][:,None]).T

        #R
        dL_dR,_ = linalg.dtrtrs(R, -dL_dA_via_v.T.dot(A), trans=1, lower=1)
        dL_dR -= A.T.dot(dF_dmu).dot(RiTm.T).T

        #backprop dL_dR for dL_dKmm
        dL_dKmm = choleskies.backprop_gradient(dL_dR, R)

        dL_dKdiag = dF_dv.sum(1)

        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum() - KL

        dL_dchol = choleskies.triang_to_flat(dL_dL)

        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dL_dKdiag, 'dL_dm':dL_dmv, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}

        #get the posterior in terms of u for GPy compat.
        q_u_mean = np.dot(R, q_v_mean)

        Su = Sv.copy()
        for i in range(num_outputs):
            Su[i] = np.dot(R, Sv[i]).dot(R.T)


        return Posterior(mean=q_u_mean, cov=Su.T, K=Kmm, prior_mean=0.), log_marginal, grad_dict
