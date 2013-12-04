import numpy as np

class Posterior(object):
    """
    An object to represent a Gaussian posterior over latent function values. 
    """
    def __init__(self, log_marginal, dL_dmean=None, cov=None, prec=None):
        self._log_marginal = log_marginal

        #TODO: accept the init arguments, make sure we've got enough information to compute everything.

    @property
    def mean(self):
        if self._mean is None:
            self._mean = ??
        return self._mean

    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = ??
        return self._covariance

    @property
    def precision(self):
        if self._precision is None:
            self._precision = ??
        return self._precision

    @prop




