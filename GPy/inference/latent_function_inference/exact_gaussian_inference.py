# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from posterior import Posterior
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

    def get_YYTfactor(self, Y):
        """
        find a matrix L which satisfies LL^T = YY^T.

        Note that L may have fewer columns than Y, else L=Y.
        """
        N, D = Y.shape
        if (N>D):
            return Y
        else:
            #if Y in self.cache, return self.Cache[Y], else store Y in cache and return L.
            #print "WARNING: N>D of Y, we need caching of L, such that L*L^T = Y, returning Y still!"
            return Y

    def inference(self, kern, X, likelihood, Y, Y_metadata=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """
        YYT_factor = self.get_YYTfactor(Y)

        K = kern.K(X)

        Ky = K.copy()
        diag.add(Ky, likelihood.gaussian_variance(Y_metadata))
        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK),Y_metadata)

        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}
