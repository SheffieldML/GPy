'''
Created on 6 Nov 2013

@author: maxz
'''

import numpy as np
from parameterized import Parameterized
from param import Param
from transformations import Logexp, Logistic,__fixed__
from GPy.util.misc import param_to_array
from GPy.util.caching import Cache_this

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
    def __init__(self, pi=None, learnPi=False, variance = 1.0, name='SpikeAndSlabPrior', **kw):
        super(SpikeAndSlabPrior, self).__init__(name=name, **kw)        
        self.variance = Param('variance',variance)
        self.learnPi = learnPi
        if learnPi:
            self.pi = Param('Pi', pi, Logistic(1e-10,1.-1e-10))
        else:
            self.pi = Param('Pi', pi, __fixed__)
        self.link_parameter(self.pi)


    def KL_divergence(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma,gamma1 = variational_posterior.gamma_probabilities()
        log_gamma,log_gamma1 = variational_posterior.gamma_log_prob()
        if len(self.pi.shape)==2:
            idx = np.unique(gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi
            
        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        var_gamma = (gamma*(log_gamma-np.log(pi))).sum()+(gamma1*(log_gamma1-np.log(1-pi))).sum()
        return var_gamma+ (gamma* (np.log(self.variance)-1. +var_mean + var_S)).sum()/2.

    def update_gradients_KL(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        gamma,gamma1 = variational_posterior.gamma_probabilities()
        log_gamma,log_gamma1 = variational_posterior.gamma_log_prob()
        if len(self.pi.shape)==2:
            idx = np.unique(gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        variational_posterior.binary_prob.gradient -= (np.log((1-pi)/pi)+log_gamma-log_gamma1+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.)*gamma*gamma1
        mu.gradient -= gamma*mu/self.variance
        S.gradient -= (1./self.variance - 1./S) * gamma /2.
        if self.learnPi:
            if len(self.pi)==1:
                self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum()
            elif len(self.pi.shape)==1:
                self.pi.gradient = (gamma/self.pi - (1.-gamma)/(1.-self.pi)).sum(axis=0)
            else:
                self.pi[idx].gradient = (gamma/self.pi[idx] - (1.-gamma)/(1.-self.pi[idx]))

class VariationalPosterior(Parameterized):
    def __init__(self, means=None, variances=None, name='latent space', *a, **kw):
        super(VariationalPosterior, self).__init__(name=name, *a, **kw)
        self.mean = Param("mean", means)
        self.variance = Param("variance", variances, Logexp())
        self.ndim = self.mean.ndim
        self.shape = self.mean.shape
        self.num_data, self.input_dim = self.mean.shape
        self.link_parameters(self.mean, self.variance)
        self.num_data, self.input_dim = self.mean.shape
        if self.has_uncertain_inputs():
            assert self.variance.shape == self.mean.shape, "need one variance per sample and dimenion"

    def set_gradients(self, grad):
        self.mean.gradient, self.variance.gradient = grad

    def _raveled_index(self):
        index = np.empty(dtype=int, shape=0)
        size = 0
        for p in self.parameters:
            index = np.hstack((index, p._raveled_index()+size))
            size += p._realsize_ if hasattr(p, '_realsize_') else p.size
        return index

    def has_uncertain_inputs(self):
        return not self.variance is None

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size
            n.size = n.mean.size + n.variance.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(VariationalPosterior, self).__getitem__(s)

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
        import matplotlib
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
        self.gamma = Param("binary_prob",binary_prob)
        self.link_parameter(self.gamma)
        
    @Cache_this(limit=5)
    def gamma_probabilities(self):
        prob = np.zeros_like(param_to_array(self.gamma))
        prob[self.gamma>-710] = 1./(1.+np.exp(-self.gamma[self.gamma>-710]))
        prob1 = -np.zeros_like(param_to_array(self.gamma))
        prob1[self.gamma<710] = 1./(1.+np.exp(self.gamma[self.gamma<710]))
        return prob, prob1
    
    @Cache_this(limit=5)
    def gamma_log_prob(self):
        loggamma = param_to_array(self.gamma).copy()
        loggamma[loggamma>-40] = -np.log1p(np.exp(-loggamma[loggamma>-40]))
        loggamma1 = -param_to_array(self.gamma).copy()
        loggamma1[loggamma1>-40] = -np.log1p(np.exp(-loggamma1[loggamma1>-40]))
        return loggamma,loggamma1

    def set_gradients(self, grad):
        self.mean.gradient, self.variance.gradient, self.gamma.gradient = grad

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['binary_prob'] = self.binary_prob[s]
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n.parameters[dc['binary_prob']._parent_index_] = dc['binary_prob']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size
            n.size = n.mean.size + n.variance.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(VariationalPrior, self).__getitem__(s)

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot_SpikeSlab(self,*args, **kwargs)
