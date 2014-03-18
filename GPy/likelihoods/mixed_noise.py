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

    def gaussian_variance(self, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        variance = np.zeros(ind.size)
        for lik, j in zip(self.likelihoods_list, range(len(self.likelihoods_list))):
            variance[ind==j] = lik.variance
        return variance[:,None]

    def betaY(self,Y,Y_metadata):
        return Y/self.gaussian_variance(Y_metadata=Y_metadata)

    def update_gradients(self, gradients):
        self.gradient = gradients

    def exact_inference_gradients(self, dL_dKdiag, Y_metadata):
        assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        ind = Y_metadata['output_index'].flatten()
        return np.array([dL_dKdiag[ind==i].sum() for i in range(len(self.likelihoods_list))])

    def predictive_values(self, mu, var, full_cov=False, Y_metadata=None):
        if all([isinstance(l, Gaussian) for l in self.likelihoods_list]):
            ind = Y_metadata['output_index'].flatten()
            _variance = np.array([self.likelihoods_list[j].variance for j in ind ])
            if full_cov:
                var += np.eye(var.shape[0])*_variance
            else:
                var += _variance
            return mu, var
        else:
            raise NotImplementedError

    def predictive_variance(self, mu, sigma, **other_shit):
        if isinstance(noise_index,int):
            _variance = self.variance[noise_index]
        else:
            _variance = np.array([ self.variance[j] for j in noise_index ])[:,None]
        return _variance + sigma**2


    def covariance_matrix(self, Y, Y_metadata):
        #assert all([isinstance(l, Gaussian) for l in self.likelihoods_list])
        #ind = Y_metadata['output_index'].flatten()
        #variance = np.zeros(Y.shape[0])
        #for lik, j in zip(self.likelihoods_list, range(len(self.likelihoods_list))):
        #    variance[ind==j] = lik.variance
        #return np.diag(variance)
        return np.diag(self.gaussian_variance(Y_metadata).flatten())


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

