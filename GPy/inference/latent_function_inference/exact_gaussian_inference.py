# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .posterior import PosteriorExact as Posterior
from ...util.linalg import pdinv, dpotrs, tdot
from ...util import diag
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class ExactGaussianInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, precision=None, Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if precision is None:
            precision = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y-m

        if K is None:
            K = kern.K(X)

        Ky = K.copy()
        diag.add(Ky, precision+1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO
