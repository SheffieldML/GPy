# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
from .../util.linalg import pdinv, dpotrs, tdot
log_2_pi = np.log(2*np.pi)


class exact_gaussian_inference(object):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        self._YYTfactor_cache = caching.cache()

     def get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LLT = YYT. 

        Note that L may have fewer columns than Y.
        """
        N, D = Y.shape
        if (N>D):
            return Y
        else:
            #if Y in self.cache, return self.Cache[Y], else stor Y in cache and return L.
            raise NotImplementedError, 'TODO' #TODO

    def inference(self, K, likelihood, Y, Y_metadata=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """
        YYT_factor = self.get_YYTfactor(Y)

        Wi, LW, LWi, W_logdet = pdinv(K + likelhood.covariance(Y, Y_metadata))

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor.T)

        dL_dtheta_lik = likelihood.dL_dtheta(np.diag(dL_dK))

        return Posterior(log_marginal, dL_DK, dL_dtheta_lik, LW, alpha, K):


