# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from ...util.linalg import pdinv
from .posterior import Posterior
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class VarGauss(LatentFunctionInference):
    """
    The Variational Gaussian Approximation revisited

    @article{Opper:2009,
        title = {The Variational Gaussian Approximation Revisited},
        author = {Opper, Manfred and Archambeau, C{\'e}dric},
        journal = {Neural Comput.},
        year = {2009},
        pages = {786--792},
    }
    """
    def __init__(self, alpha, beta):
        """
        :param alpha: GPy.core.Param varational parameter
        :param beta: GPy.core.Param varational parameter
        """
        self.alpha, self.beta = alpha, beta

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, Z=None):
        if mean_function is not None:
            raise NotImplementedError
        num_data, output_dim = Y.shape
        assert output_dim ==1, "Only one output supported"

        K = kern.K(X)
        m = K.dot(self.alpha)
        KB = K*self.beta[:, None]
        BKB = KB*self.beta[None, :]
        A = np.eye(num_data) + BKB
        Ai, LA, _, Alogdet = pdinv(A)
        Sigma = np.diag(self.beta**-2) - Ai/self.beta[:, None]/self.beta[None, :]  # posterior coavairance: need full matrix for gradients
        var = np.diag(Sigma).reshape(-1,1)

        F, dF_dm, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, m, var, Y_metadata=Y_metadata)
        if dF_dthetaL is not None:
            dL_dthetaL = dF_dthetaL.sum(1).sum(1)
        else:
            dL_dthetaL = np.array([])
        dF_da = np.dot(K, dF_dm)
        SigmaB = Sigma*self.beta
        #dF_db_ = -np.diag(Sigma.dot(np.diag(dF_dv.flatten())).dot(SigmaB))*2
        dF_db = -2*np.sum(Sigma**2 * (dF_dv * self.beta), 0)
        #assert np.allclose(dF_db, dF_db_)

        KL = 0.5*(Alogdet + np.trace(Ai) - num_data + np.sum(m*self.alpha))
        dKL_da = m
        A_A2 = Ai - Ai.dot(Ai)
        dKL_db = np.diag(np.dot(KB.T, A_A2))
        log_marginal = F.sum() - KL
        self.alpha.gradient = dF_da - dKL_da
        self.beta.gradient = dF_db - dKL_db

        # K-gradients
        dKL_dK = 0.5*(self.alpha*self.alpha.T + self.beta[:, None]*self.beta[None, :]*A_A2)
        tmp = Ai*self.beta[:, None]/self.beta[None, :]
        dF_dK = self.alpha*dF_dm.T + np.dot(tmp*dF_dv, tmp.T)

        return Posterior(mean=m, cov=Sigma ,K=K),\
               log_marginal,\
               {'dL_dK':dF_dK-dKL_dK, 'dL_dthetaL':dL_dthetaL}
