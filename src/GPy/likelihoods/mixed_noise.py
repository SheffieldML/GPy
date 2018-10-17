# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from . import link_functions
from .likelihood import Likelihood
from .gaussian import Gaussian
from ..core.parameterization import Param
from paramz.transformations import Logexp
from ..core.parameterization import Parameterized
import itertools

class MixedNoise(Likelihood):
    def __init__(self, likelihoods_list, name='mixed_noise'):
        #NOTE at the moment this likelihood only works for using a list of gaussians
        super(Likelihood, self).__init__(name=name)

        self.link_parameters(*likelihoods_list)
        self.likelihoods_list = likelihoods_list
        self.log_concave = False

    def gaussian_variance(self, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        variance = np.zeros(ind.size)
        for lik, j in zip(self.likelihoods_list, range(len(self.likelihoods_list))):
            variance[ind==j] = lik.variance
        return variance

    def betaY(self,Y,Y_metadata):
        #TODO not here.
        return Y/self.gaussian_variance(Y_metadata=Y_metadata)[:,None]

    def update_gradients(self, gradients):
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        return np.array([dL_dKdiag[ind==i].sum() for i in range(len(self.likelihoods_list))])

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        ind = Y_metadata['output_index'].flatten()
        _variance = np.array([self.likelihoods_list[j].variance for j in ind ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
        else:
            var += _variance
        return mu, var

    def predictive_variance(self, mu, sigma, Y_metadata):
        _variance = self.gaussian_variance(Y_metadata)
        return _variance + sigma**2

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata):
        ind = Y_metadata['output_index'].flatten()
        outputs = np.unique(ind)
        Q = np.zeros( (mu.size,len(quantiles)) )
        for j in outputs:
            q = self.likelihoods_list[j].predictive_quantiles(mu[ind==j,:],
                var[ind==j,:],quantiles,Y_metadata=None)
            Q[ind==j,:] = np.hstack(q)
        return [q[:,None] for q in Q.T]

    def samples(self, gp, Y_metadata):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        N1, N2 = gp.shape
        Ysim = np.zeros((N1,N2))
        ind = Y_metadata['output_index'].flatten()
        for j in np.unique(ind):
            flt = ind==j
            gp_filtered = gp[flt,:]
            n1 = gp_filtered.shape[0]
            lik = self.likelihoods_list[j]
            _ysim = np.array([np.random.normal(lik.gp_link.transf(gpj), scale=np.sqrt(lik.variance), size=1) for gpj in gp_filtered.flatten()])
            Ysim[flt,:] = _ysim.reshape(n1,N2)
        return Ysim
