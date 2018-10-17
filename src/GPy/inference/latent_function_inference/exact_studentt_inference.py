# Copyright (c) 2017, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from . import LatentFunctionInference
from .posterior import StudentTPosterior
from ...util.linalg import pdinv, dpotrs, tdot
from ...util import diag

import numpy as np
from scipy.special import gammaln, digamma


class ExactStudentTInference(LatentFunctionInference):
    """
    An object for inference of student-t processes (not for GP with student-t likelihood!).

    The function self.inference returns a StudentTPosterior object, which summarizes
    the posterior.
    """

    def inference(self, kern, X, Y, nu, mean_function=None, K=None):
        m = 0 if mean_function is None else mean_function.f(X)
        K = kern.K(X) if K is None else K

        YYT_factor = Y - m
        Ky = K.copy()
        diag.add(Ky, 1e-8)

        # Posterior representation
        Wi, LW, LWi, W_logdet = pdinv(Ky)
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        beta = np.sum(alpha * YYT_factor)
        posterior = StudentTPosterior(nu, woodbury_chol=LW, woodbury_vector=alpha, K=K)

        # Log marginal
        N = Y.shape[0]
        D = Y.shape[1]
        log_marginal = 0.5 * (-N * np.log((nu - 2) * np.pi) - W_logdet - (nu + N) * np.log(1 + beta / (nu - 2)))
        log_marginal += gammaln((nu + N) / 2) - gammaln(nu / 2)

        # Gradients
        dL_dK = 0.5 * ((nu + N) / (nu + beta - 2) * tdot(alpha) - D * Wi)
        dL_dnu = -N / (nu - 2.) + digamma(0.5 * (nu + N)) - digamma(0.5 * nu)
        dL_dnu -= np.log(1 + beta / (nu - 2.))
        dL_dnu += ((nu + N) * beta) / ((nu - 2) * (beta + nu - 2))
        dL_dnu *= 0.5
        gradients = {'dL_dK': dL_dK, 'dL_dnu': dL_dnu, 'dL_dm': alpha}

        return posterior, log_marginal, gradients
