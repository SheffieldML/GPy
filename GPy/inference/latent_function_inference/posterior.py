# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

class Posterior(object):
    """
    An object to represent a Gaussian posterior over latent function values.
    This may be computed exactly for Gaussian likelihoods, or approximated for
    non-Gaussian likelihoods. 

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes. 

    """
    def __init__(self, log_marginal, dL_dK, dL_dtheta_lik, woodbury_chol, woodbury_vector, K):
        """
        log_marginal: log p(Y|X)
        DL_dK: d/dK log p(Y|X)
        dL_dtheta_lik : d/dtheta log p(Y|X) (where theta are the parameters of the likelihood)
        woodbury_chol : a lower triangular matrix L that satisfies posterior_covariance = K - K L^{-T} L^{-1} K
        woodbury_vector : a matrix (or vector, as Nx1 matrix) M which satisfies posterior_mean = K M
        K : the proir covariance (required for lazy computation of various quantities)
        """
        self.log_marginal = log_marginal
        self.dL_dK = dL_dK
        self.dL_dtheta_lik = dL_dtheta_lik
        self._woodbury_chol = woodbury_chol
        self._woodbury_vector = woodbury_vector
        self._K = K

        #these are computed lazily below
        self._mean = None
        self._covariance = None
        self._precision = None

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.dot(self._K, self._woodbury_vector)
        return self._mean

    @property
    def covariance(self):
        if self._covariance is None:
            LiK, _ = dpotrs(self._woodbury_chol, self._K)
            self._covariance = self._K - tdot(LiK.T)
        return self._covariance

    @property
    def precision(self):
        if self._precision is None:
            self._precision = np.linalg.inv(self.covariance)
        return self._precision





