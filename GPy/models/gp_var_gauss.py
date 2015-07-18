# Copyright (c) 2014, James Hensman, Alan Saul
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
from scipy import stats
from scipy.special import erf
from ..core.model import Model
from ..core.parameterization import ObsAr
from .. import kern
from ..core.parameterization.param import Param
from ..util.linalg import pdinv

log_2_pi = np.log(2*np.pi)


class GPVariationalGaussianApproximation(Model):
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
    def __init__(self, X, Y, kernel, likelihood,Y_metadata=None):
        Model.__init__(self,'Variational GP classification')
        # accept the construction arguments
        self.X = ObsAr(X)
        self.Y = Y
        self.num_data, self.input_dim = self.X.shape
        self.Y_metadata = Y_metadata

        self.kern = kernel
        self.likelihood = likelihood
        self.link_parameter(self.kern)
        self.link_parameter(self.likelihood)

        self.alpha = Param('alpha', np.zeros((self.num_data,1))) # only one latent fn for now.
        self.beta = Param('beta', np.ones(self.num_data))
        self.link_parameter(self.alpha)
        self.link_parameter(self.beta)

    def log_likelihood(self):
        return self._log_lik

    def parameters_changed(self):
        K = self.kern.K(self.X)
        m = K.dot(self.alpha)
        KB = K*self.beta[:, None]
        BKB = KB*self.beta[None, :]
        A = np.eye(self.num_data) + BKB
        Ai, LA, _, Alogdet = pdinv(A)
        Sigma = np.diag(self.beta**-2) - Ai/self.beta[:, None]/self.beta[None, :]  # posterior coavairance: need full matrix for gradients
        var = np.diag(Sigma).reshape(-1,1)

        F, dF_dm, dF_dv, dF_dthetaL = self.likelihood.variational_expectations(self.Y, m, var, Y_metadata=self.Y_metadata)
        self.likelihood.gradient = dF_dthetaL.sum(1).sum(1)
        dF_da = np.dot(K, dF_dm)
        SigmaB = Sigma*self.beta
        dF_db = -np.diag(Sigma.dot(np.diag(dF_dv.flatten())).dot(SigmaB))*2
        KL = 0.5*(Alogdet + np.trace(Ai) - self.num_data + np.sum(m*self.alpha))
        dKL_da = m
        A_A2 = Ai - Ai.dot(Ai)
        dKL_db = np.diag(np.dot(KB.T, A_A2))
        self._log_lik = F.sum() - KL
        self.alpha.gradient = dF_da - dKL_da
        self.beta.gradient = dF_db - dKL_db

        # K-gradients
        dKL_dK = 0.5*(self.alpha*self.alpha.T + self.beta[:, None]*self.beta[None, :]*A_A2)
        tmp = Ai*self.beta[:, None]/self.beta[None, :]
        dF_dK = self.alpha*dF_dm.T + np.dot(tmp*dF_dv, tmp.T)
        self.kern.update_gradients_full(dF_dK - dKL_dK, self.X)

    def _raw_predict(self, Xnew):
        """
        Predict the function(s) at the new point(s) Xnew.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        """
        Wi, _, _, _ = pdinv(self.kern.K(self.X) + np.diag(self.beta**-2))
        Kux = self.kern.K(self.X, Xnew)
        mu = np.dot(Kux.T, self.alpha)
        WiKux = np.dot(Wi, Kux)
        Kxx = self.kern.Kdiag(Xnew)
        var = Kxx - np.sum(WiKux*Kux, 0)

        return mu, var.reshape(-1,1)
