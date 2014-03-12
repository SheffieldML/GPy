import numpy as np
from scipy import stats, special
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
import link_functions
from likelihood import Likelihood
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from ..core.parameterization import Parameterized
import itertools

class MixedNoise(Likelihood):
    def __init__(self, likelihoods_list, noise_index, variance = None, name='mixed_noise'):

        Nlike = len(likelihoods_list)
        self.order = np.unique(noise_index)

        assert self.order.size == Nlike

        if variance is None:
            variance = np.ones(Nlike)
        else:
            assert variance.size == Nlike

        super(Likelihood, self).__init__(name=name)

        self.add_parameters(*likelihoods_list)
        self.likelihoods_list = likelihoods_list
        self.noise_index = noise_index
        self.log_concave = False
        self.likelihoods_indices = [noise_index.flatten()==j for j in self.order]

    def covariance_matrix(self, Y, noise_index, **Y_metadata):
        variance = np.zeros(Y.shape[0])
        for lik, ind in itertools.izip(self.likelihoods_list, self.likelihoods_indices):
            variance[ind] = lik.variance
        return np.diag(variance)

    def update_gradients(self, partial, noise_index, **Y_metadata):
        [lik.update_gradients(partial[ind]) for lik,ind in itertools.izip(self.likelihoods_list, self.likelihoods_indices)]

    def predictive_values(self, mu, var, full_cov=False, noise_index=None, **Y_metadata):
        _variance = np.array([ self.likelihoods_list[j].variance for j in noise_index ])
        if full_cov:
            var += np.eye(var.shape[0])*_variance
            d = 2*np.sqrt(np.diag(var))
            low, up = mu - d, mu + d
        else:
            var += _variance
            d = 2*np.sqrt(var)
            low, up = mu - d, mu + d
        return mu, var, low, up

    def predictive_variance(self, mu, sigma, noise_index, predictive_mean=None, **Y_metadata):
        if isinstance(noise_index,int):
            _variance = self.variance[noise_index]
        else:
            _variance = np.array([ self.variance[j] for j in noise_index ])[:,None]
        return _variance + sigma**2
