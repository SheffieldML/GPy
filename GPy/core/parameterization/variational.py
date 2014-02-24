'''
Created on 6 Nov 2013

@author: maxz
'''

import numpy as np
from parameterized import Parameterized
from param import Param
from transformations import Logexp

class VariationalPrior(object):
    def KL_divergence(self, variational_posterior):
        raise NotImplementedError, "override this for variational inference of latent space"

    def update_gradients_KL(self, variational_posterior):
        """
        updates the gradients for mean and variance **in place**
        """
        raise NotImplementedError, "override this for variational inference of latent space"
    
class NormalPrior(VariationalPrior):
    def KL_divergence(self, variational_posterior):
        var_mean = np.square(variational_posterior.mean).sum()
        var_S = (variational_posterior.variance - np.log(variational_posterior.variance)).sum()
        return 0.5 * (var_mean + var_S) - 0.5 * variational_posterior.input_dim * variational_posterior.num_data

    def update_gradients_KL(self, variational_posterior):
        # dL:
        variational_posterior.mean.gradient -= variational_posterior.mean
        variational_posterior.variance.gradient -= (1. - (1. / (variational_posterior.variance))) * 0.5


class VariationalPosterior(Parameterized):
    def __init__(self, means=None, variances=None, name=None, **kw):
        super(VariationalPosterior, self).__init__(name=name, **kw)
        self.mean = Param("mean", means)
        self.ndim = self.mean.ndim
        self.shape = self.mean.shape
        self.variance = Param("variance", variances, Logexp())
        self.add_parameters(self.mean, self.variance)
        self.num_data, self.input_dim = self.mean.shape
        if self.has_uncertain_inputs():
            assert self.variance.shape == self.mean.shape, "need one variance per sample and dimenion"
    
    def has_uncertain_inputs(self):
        return not self.variance is None


class NormalPosterior(VariationalPosterior):
    '''
    NormalPosterior distribution for variational approximations.

    holds the means and variances for a factorizing multivariate normal distribution
    '''

    def plot(self, *args):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self,*args)

class SpikeAndSlab(VariationalPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        super(SpikeAndSlab, self).__init__(means, variances, name)
        self.gamma = Param("binary_prob",binary_prob,)
        self.add_parameter(self.gamma)

    def plot(self, *args):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self,*args)
