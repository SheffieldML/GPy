# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import numpy as np

class SsmPosterior(object):
    """
    Specially intended for the SSM Regression case
    An object to represent a Gaussian posterior over latent function values, p(f|D).

    The purpose of this class is to serve as an interface between the inference
    schemes and the model classes.

    """
    def __init__(self, mu_f = None, V_f=None, mu_s=None, V_s=None, expectations=None):
        """
        mu_f : mean values predicted during kalman filtering step
        var_f : variance predicted during the kalman filtering step
        mu_s : mean values predicted during backwards smoothing step
        var_s : variance predicted during backwards smoothing step
        expectations : posterior expectations
        """

        if ((mu_f is not None) and (V_f is not None) and
            (mu_s is not None) and (V_s is not None) and 
            (expectations is not None)):
            pass # we have sufficient to compute the posterior
        else:
            raise ValueError("insufficient information to compute predictions")

        self._mu_f = mu_f
        self._V_f = V_f
        self._mu_s = mu_s
        self._V_s = V_s
        self._expectations = expectations

    @property
    def mu_f(self):
        """
        Mean values predicted during kalman filtering step mean
        """
        return self._mu_f

    @property
    def V_f(self):
        """
        Variance predicted during the kalman filtering step
        """
        return self._V_f

    @property
    def mu_s(self):
        """
        Mean values predicted during kalman backwards smoothin mean
        """
        return self._mu_s

    @property
    def V_s(self):
        """
        Variance predicted during backwards smoothing step
        """
        return self._V_s

    @property
    def expectations(self):
        """
        Posterior expectations
        """
        return self._expectations
    
