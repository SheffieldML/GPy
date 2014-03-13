import numpy as np
from scipy import stats, special
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
import link_functions
from likelihood import Likelihood
from gaussian import Gaussian
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp
from ..core.parameterization import Parameterized
import itertools

class MixedNoise(Likelihood):
    def __init__(self, likelihoods_list, name='mixed_noise'):

        super(Likelihood, self).__init__(name=name)

        self.add_parameters(*likelihoods_list)
        self.likelihoods_list = likelihoods_list
        self.log_concave = False

    def update_gradients(self, gradients):
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index']
        return np.array([dL_dKdiag[ind==i].sum() for i in range(len(self.likelihoods_list))])

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        if all([isinstance(l, Gaussian) for l in self.likelihoods_list]):
            ind = Y_metadata['output_index']
            _variance = np.array([self.likelihoods_list[j].variance for j in ind ])
            if full_cov:
                var += np.eye(var.shape[0])*_variance
                d = 2*np.sqrt(np.diag(var))
                low, up = mu - d, mu + d
            else:
                var += _variance
                d = 2*np.sqrt(var)
                low, up = mu - d, mu + d
            return mu, var, low, up
        else:
            raise NotImplementedError

    def predictive_variance(self, mu, sigma, **other_shit):
        if isinstance(noise_index,int):
            _variance = self.variance[noise_index]
        else:
            _variance = np.array([ self.variance[j] for j in noise_index ])[:,None]
        return _variance + sigma**2


    def covariance_matrix(self, Y, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        variance = np.zeros(Y.shape[0])
        for lik, ind in itertools.izip(self.likelihoods_list, self.likelihoods_indices):
            variance[ind] = lik.variance
        return np.diag(variance)

