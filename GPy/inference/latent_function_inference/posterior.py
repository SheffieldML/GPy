# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ...util.linalg import pdinv, dpotrs, tdot, dtrtrs

class Posterior(object):
    """
    An object to represent a Gaussian posterior over latent function values.
    This may be computed exactly for Gaussian likelihoods, or approximated for
    non-Gaussian likelihoods. 

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes. 

    """
    def __init__(self, log_marginal, dL_dK, dL_dtheta_lik, woodbury_chol=None, woodbury_vector=None, K=None, mean=None, cov=None, K_chol=None):
        """
        log_marginal: log p(Y|X)
        dL_dK: d/dK log p(Y|X)
        dL_dtheta_lik : d/dtheta log p(Y|X) (where theta are the parameters of the likelihood)
        woodbury_chol : a lower triangular matrix L that satisfies posterior_covariance = K - K L^{-T} L^{-1} K
        woodbury_vector : a matrix (or vector, as Nx1 matrix) M which satisfies posterior_mean = K M
        K : the proir covariance (required for lazy computation of various quantities)
        mean : the posterior mean
        cov : the posterior covariance

        Not all of the above need to be supplied! You *must* supply:

          log_marginal
          dL_dK
          dL_dtheta_lik
          K (for lazy computation)

       You may supply either:
         cc
          woodbury_chol
          woodbury_vector

        Or:

          mean
          cov
          K_chol (for lazy computation)

        Of course, you can supply more than that, but this class will lazily compute all other quantites on demand.

        From the supplied quantities, all of the others will be computed on demand (lazy computation)
        """
        #obligatory
        self.log_marginal = log_marginal
        self.dL_dK = dL_dK
        self.dL_dtheta_lik = dL_dtheta_lik
        self._K = K

        if ((woodbury_chol is not None) and (woodbury_vector is not None) and (K is not None)) or ((mean is not None) and (cov is not None) and (K is not None)):
            pass # we have sufficient to compute the posterior
        else:
            raise ValueError, "insufficient onformation to compute the posterior"

        #option 1:
        self._woodbury_chol = woodbury_chol
        self._woodbury_vector = woodbury_vector

        #option 2:
        self._mean = mean
        self._covariance = cov
        self._K_chol = K_chol

        #copmute this lazily
        self._precision = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.dot(self._K, self._woodbury_vector)
        return self._mean

    @property
    def covariance(self):
        if self._covariance is None:
            LiK, _ = dtrtrs(self.woodbury_chol, self._K, lower=1)
            self._covariance = self._K - tdot(LiK.T)
        return self._covariance

    @property
    def precision(self):
        if self._precision is None:
            self._precision, _, _, _ = pdinv(self.covariance)
        return self._precision

    @property
    def woodbury_chol(self):
        if self._woodbury_chol is None:
            B = self._K - self._covariance
            tmp, _ = dpotrs(self._K_chol, B)
            Wi, _ = dpotrs(self._K_chol, tmp.T)
            _, _, self._woodbury_chol, _ = pdinv(Wi)
        return self._woodbury_chol

    @property
    def woodbury_vector(self):
        if self._woodbury_vector is None:
            self._woodbury_vector, _ = dpotrs(self._K_chol, self.mean)
        return self._woodbury_vector





