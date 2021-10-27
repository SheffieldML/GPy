'''
Created on 6 Nov 2013

@author: maxz
'''

import numpy as np
from .parameterized import Parameterized
from .param import Param
from paramz.transformations import Logexp, Logistic,__fixed__

class VariationalPrior(Parameterized):
    def __init__(self, name='latent prior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)

    def KL_divergence(self, variational_posterior):
        raise NotImplementedError("override this for variational inference of latent space")

    def update_gradients_KL(self, variational_posterior):
        """
        updates the gradients for mean and variance **in place**
        """
        raise NotImplementedError("override this for variational inference of latent space")

class NormalPrior(VariationalPrior):
    def __init__(self, name='normal_prior', **kw):
        super(VariationalPrior, self).__init__(name=name, **kw)

    def KL_divergence(self, variational_posterior):
        var_mean = np.square(variational_posterior.mean).sum()
        var_S = (variational_posterior.variance - np.log(variational_posterior.variance)).sum()
        return 0.5 * (var_mean + var_S) - 0.5 * variational_posterior.input_dim * variational_posterior.num_data

    def update_gradients_KL(self, variational_posterior):
        # dL:
        variational_posterior.mean.gradient -= variational_posterior.mean
        variational_posterior.variance.gradient -= (1. - (1. / (variational_posterior.variance))) * 0.5

class SpikeAndSlabPrior(VariationalPrior):
    def __init__(self, pi=None, learnPi=False, variance = 1.0, group_spike=False, name='SpikeAndSlabPrior', **kw):
        super(SpikeAndSlabPrior, self).__init__(name=name, **kw)
        self.group_spike = group_spike
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
        if self.group_spike:
            gamma = variational_posterior.gamma.values[0]
        else:
            gamma = variational_posterior.gamma.values
        if len(self.pi.shape)==2:
            idx = np.unique(variational_posterior.gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        var_gamma = (gamma*np.log(gamma/pi)).sum()+((1-gamma)*np.log((1-gamma)/(1-pi))).sum()
        return var_gamma+ (gamma* (np.log(self.variance)-1. +var_mean + var_S)).sum()/2.

    def update_gradients_KL(self, variational_posterior):
        mu = variational_posterior.mean
        S = variational_posterior.variance
        if self.group_spike:
            gamma = variational_posterior.gamma.values[0]
        else:
            gamma = variational_posterior.gamma.values
        if len(self.pi.shape)==2:
            idx = np.unique(variational_posterior.gamma._raveled_index()/gamma.shape[-1])
            pi = self.pi[idx]
        else:
            pi = self.pi

        if self.group_spike:
            dgamma = np.log((1-pi)/pi*gamma/(1.-gamma))/variational_posterior.num_data
        else:
            dgamma = np.log((1-pi)/pi*gamma/(1.-gamma))
        variational_posterior.binary_prob.gradient -= dgamma+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
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
            assert self.variance.shape == self.mean.shape, "need one variance per sample and dimension"

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

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self, *args, **kwargs)

    def KL(self, other):
        """Compute the KL divergence to another NormalPosterior Object. This only holds, if the two NormalPosterior objects have the same shape, as we do computational tricks for the multivariate normal KL divergence.
        """
        return .5*(
            np.sum(self.variance/other.variance)
            + ((other.mean-self.mean)**2/other.variance).sum()
            - self.num_data * self.input_dim
            + np.sum(np.log(other.variance)) - np.sum(np.log(self.variance))
            )

class SpikeAndSlabPosterior(VariationalPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, group_spike=False, sharedX=False, name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        super(SpikeAndSlabPosterior, self).__init__(means, variances, name)
        self.group_spike = group_spike
        self.sharedX = sharedX
        if sharedX:
            self.mean.fix(warning=False)
            self.variance.fix(warning=False)
        if group_spike:
            self.gamma_group = Param("binary_prob_group",binary_prob.mean(axis=0),Logistic(1e-10,1.-1e-10))
            self.gamma = Param("binary_prob",binary_prob, __fixed__)
            self.link_parameters(self.gamma_group,self.gamma)
        else:
            self.gamma = Param("binary_prob",binary_prob,Logistic(1e-10,1.-1e-10))
            self.link_parameter(self.gamma)

    def propogate_val(self):
        if self.group_spike:
            self.gamma.values[:] = self.gamma_group.values

    def collate_gradient(self):
        if self.group_spike:
            self.gamma_group.gradient = self.gamma.gradient.reshape(self.gamma.shape).sum(axis=0)

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
            oversize = self.size - self.mean.size - self.variance.size - self.gamma.size
            n.size = n.mean.size + n.variance.size + n.gamma.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(SpikeAndSlabPosterior, self).__getitem__(s)

    def plot(self, *args, **kwargs):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot_SpikeSlab(self,*args, **kwargs)
