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
    The Variational Gaussian Approximation revisited implementation for regression

    @article{Opper:2009,
        title = {The Variational Gaussian Approximation Revisited},
        author = {Opper, Manfred and Archambeau, C{\'e}dric},
        journal = {Neural Comput.},
        year = {2009},
        pages = {786--792},
    }
    """
    def __init__(self, X, Y, kernel=None):
        Model.__init__(self,'Variational GP classification')
        # accept the construction arguments
        self.X = ObsAr(X)
        if kernel is None:
            kernel = kern.RBF(X.shape[1]) + kern.White(X.shape[1], 0.01)
        self.kern = kernel
        self.link_parameter(self.kern)
        self.num_data, self.input_dim = self.X.shape

        self.alpha = Param('alpha', np.zeros(self.num_data))
        self.beta = Param('beta', np.ones(self.num_data))
        self.link_parameter(self.alpha)
        self.link_parameter(self.beta)

        self.gh_x, self.gh_w = np.polynomial.hermite.hermgauss(20)
        self.Ysign = np.where(Y==1, 1, -1).flatten()

    def log_likelihood(self):
        """
        Marginal log likelihood evaluation
        """
        return self._log_lik

    def likelihood_quadrature(self, m, v):
        """
        Perform Gauss-Hermite quadrature over the log of the likelihood, with a fixed weight
        """
        # assume probit for now.
        X = self.gh_x[None, :]*np.sqrt(2.*v[:, None]) + (m*self.Ysign)[:, None]
        p = stats.norm.cdf(X)
        N = stats.norm.pdf(X)
        F = np.log(p).dot(self.gh_w)
        NoverP = N/p
        dF_dm = (NoverP*self.Ysign[:,None]).dot(self.gh_w)
        dF_dv = -0.5*(NoverP**2 + NoverP*X).dot(self.gh_w)
        return F, dF_dm, dF_dv

    def parameters_changed(self):
        K = self.kern.K(self.X)
        m = K.dot(self.alpha)
        KB = K*self.beta[:, None]
        BKB = KB*self.beta[None, :]
        A = np.eye(self.num_data) + BKB
        Ai, LA, _, Alogdet = pdinv(A)
        Sigma = np.diag(self.beta**-2) - Ai/self.beta[:, None]/self.beta[None, :]  # posterior coavairance: need full matrix for gradients
        var = np.diag(Sigma)

        F, dF_dm, dF_dv = self.likelihood_quadrature(m, var)
        dF_da = np.dot(K, dF_dm)
        SigmaB = Sigma*self.beta
        dF_db = -np.diag(Sigma.dot(np.diag(dF_dv)).dot(SigmaB))*2
        KL = 0.5*(Alogdet + np.trace(Ai) - self.num_data + m.dot(self.alpha))
        dKL_da = m
        A_A2 = Ai - Ai.dot(Ai)
        dKL_db = np.diag(np.dot(KB.T, A_A2))
        self._log_lik = F.sum() - KL
        self.alpha.gradient = dF_da - dKL_da
        self.beta.gradient = dF_db - dKL_db

        # K-gradients
        dKL_dK = 0.5*(self.alpha[None, :]*self.alpha[:, None] + self.beta[:, None]*self.beta[None, :]*A_A2)
        tmp = Ai*self.beta[:, None]/self.beta[None, :]
        dF_dK = self.alpha[:, None]*dF_dm[None, :] + np.dot(tmp*dF_dv, tmp.T)
        self.kern.update_gradients_full(dF_dK - dKL_dK, self.X)

    def predict(self, Xnew):
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

        return 0.5*(1+erf(mu/np.sqrt(2.*(var+1))))
