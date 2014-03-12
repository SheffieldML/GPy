'''
Created on 6 Nov 2013

@author: maxz
'''

import numpy as np
from parameterized import Parameterized
from param import Param
from transformations import Logexp, Logistic

class VariationalPrior(Parameterized):
    def __init__(self, name='latent space', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)
        
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

class SpikeAndSlabPrior(VariationalPrior):
    def __init__(self, pi, variance = 1.0, name='SpikeAndSlabPrior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)
        assert variance==1.0, "Not Implemented!"
        self.pi = Param('pi', pi, Logistic(1e-10,1.-1e-10))
        self.variance = Param('variance',variance)
        self.add_parameters(self.pi)
        
    def KL_divergence(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob
        var_mean = np.square(mu)
        var_S = (S - np.log(S))
        var_gamma = (gamma*np.log(gamma/self.pi)).sum()+((1-gamma)*np.log((1-gamma)/(1-self.pi))).sum()
        return var_gamma+ 0.5 * (gamma* (var_mean + var_S -1)).sum()
    
    def update_gradients_KL(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma = variational_posterior.binary_prob

        gamma.gradient -= np.log((1-self.pi)/self.pi*gamma/(1.-gamma))+(np.square(mu)+S-np.log(S)-1.)/2.
        mu.gradient -= gamma*mu
        S.gradient -= (1. - (1. / (S))) * gamma /2.
        self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum(axis=0)
        


class VariationalPosterior(Parameterized):
    def __init__(self, means=None, variances=None, name=None, *a, **kw):
        super(VariationalPosterior, self).__init__(name=name, *a, **kw)
        self.mean = Param("mean", means)
        self.variance = Param("variance", variances, Logexp())
        self.ndim = self.mean.ndim
        self.shape = self.mean.shape
        self.num_data, self.input_dim = self.mean.shape
        self.add_parameters(self.mean, self.variance)
        self.num_data, self.input_dim = self.mean.shape
        if self.has_uncertain_inputs():
            assert self.variance.shape == self.mean.shape, "need one variance per sample and dimenion"
    
    def has_uncertain_inputs(self):
        return not self.variance is None

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['_parameters_'] = copy.copy(self._parameters_)
            n.__dict__.update(dc)
            n._parameters_[dc['mean']._parent_index_] = dc['mean']
            n._parameters_[dc['variance']._parent_index_] = dc['variance']
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(VariationalPrior, self).__getitem__(s)

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

class SpikeAndSlabPosterior(VariationalPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        super(SpikeAndSlabPosterior, self).__init__(means, variances, name)
        self.gamma = Param("binary_prob",binary_prob, Logistic(1e-10,1.-1e-10))
        self.add_parameter(self.gamma)

    def plot(self, *args):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot_SpikeSlab(self,*args)
