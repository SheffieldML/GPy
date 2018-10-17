# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import kern
from ..core.parameterization import Param
from ..likelihoods import Gaussian
from GPy.core.parameterization.variational import SpikeAndSlabPrior, SpikeAndSlabPosterior,VariationalPrior
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch
from ..kern.src.psi_comp.ssrbf_psi_gpucomp import PSICOMP_SSRBF_GPU

class IBPPosterior(SpikeAndSlabPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, tau=None,  sharedX=False, name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        from paramz.transformations import Logexp
        super(IBPPosterior, self).__init__(means, variances, binary_prob, group_spike=True, name=name)
        self.sharedX = sharedX
        if sharedX:
            self.mean.fix(warning=False)
            self.variance.fix(warning=False)
        self.tau = Param("tau_", np.ones((self.gamma_group.shape[0],2)), Logexp())
        self.link_parameter(self.tau)

    def set_gradients(self, grad):
        self.mean.gradient, self.variance.gradient, self.gamma.gradient, self.tau.gradient = grad

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['binary_prob'] = self.binary_prob[s]
            dc['tau'] = self.tau
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n.parameters[dc['binary_prob']._parent_index_] = dc['binary_prob']
            n.parameters[dc['tau']._parent_index_] = dc['tau']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size - self.gamma.size - self.tau.size
            n.size = n.mean.size + n.variance.size + n.gamma.size+ n.tau.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(IBPPosterior, self).__getitem__(s)

class IBPPrior(VariationalPrior):
    def __init__(self, input_dim, alpha =2., name='IBPPrior', **kw):
        super(IBPPrior, self).__init__(name=name, **kw)
        from paramz.transformations import Logexp, __fixed__  
        self.input_dim = input_dim
        self.variance = 1.
        self.alpha = Param('alpha', alpha, __fixed__)
        self.link_parameter(self.alpha)

    def KL_divergence(self, variational_posterior):
        mu, S, gamma, tau = variational_posterior.mean.values, variational_posterior.variance.values, variational_posterior.gamma_group.values, variational_posterior.tau.values
            
        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        part1 = (gamma* (np.log(self.variance)-1. +var_mean + var_S)).sum()/2.
        
        ad = self.alpha/self.input_dim
        from scipy.special import betaln,digamma
        part2 = (gamma*np.log(gamma)).sum() + ((1.-gamma)*np.log(1.-gamma)).sum() + betaln(ad,1.)*self.input_dim \
                -betaln(tau[:,0], tau[:,1]).sum() + ((tau[:,0]-gamma-ad)*digamma(tau[:,0])).sum() + \
                ((tau[:,1]+gamma-2.)*digamma(tau[:,1])).sum() + ((2.+ad-tau[:,0]-tau[:,1])*digamma(tau.sum(axis=1))).sum()
        
        return part1+part2

    def update_gradients_KL(self, variational_posterior):
        mu, S, gamma, tau = variational_posterior.mean.values, variational_posterior.variance.values, variational_posterior.gamma_group.values, variational_posterior.tau.values

        variational_posterior.mean.gradient -= gamma*mu/self.variance
        variational_posterior.variance.gradient -= (1./self.variance - 1./S) * gamma /2.
        from scipy.special import digamma,polygamma
        dgamma = (np.log(gamma/(1.-gamma))+ digamma(tau[:,1])-digamma(tau[:,0]))/variational_posterior.num_data
        variational_posterior.binary_prob.gradient -= dgamma+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
        ad = self.alpha/self.input_dim
        common = (ad+2-tau[:,0]-tau[:,1])*polygamma(1,tau.sum(axis=1))
        variational_posterior.tau.gradient[:,0] = -((tau[:,0]-gamma-ad)*polygamma(1,tau[:,0])+common)
        variational_posterior.tau.gradient[:,1] = -((tau[:,1]+gamma-2)*polygamma(1,tau[:,1])+common)

class SLVMPosterior(SpikeAndSlabPosterior):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, tau=None,  name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        from paramz.transformations import Logexp
        super(SLVMPosterior, self).__init__(means, variances, binary_prob, group_spike=False, name=name)
        self.tau = Param("tau_", np.ones((self.gamma.shape[1],2)), Logexp())
        self.link_parameter(self.tau)

    def set_gradients(self, grad):
        self.mean.gradient, self.variance.gradient, self.gamma.gradient, self.tau.gradient = grad

    def __getitem__(self, s):
        if isinstance(s, (int, slice, tuple, list, np.ndarray)):
            import copy
            n = self.__new__(self.__class__, self.name)
            dc = self.__dict__.copy()
            dc['mean'] = self.mean[s]
            dc['variance'] = self.variance[s]
            dc['binary_prob'] = self.binary_prob[s]
            dc['tau'] = self.tau
            dc['parameters'] = copy.copy(self.parameters)
            n.__dict__.update(dc)
            n.parameters[dc['mean']._parent_index_] = dc['mean']
            n.parameters[dc['variance']._parent_index_] = dc['variance']
            n.parameters[dc['binary_prob']._parent_index_] = dc['binary_prob']
            n.parameters[dc['tau']._parent_index_] = dc['tau']
            n._gradient_array_ = None
            oversize = self.size - self.mean.size - self.variance.size - self.gamma.size - self.tau.size
            n.size = n.mean.size + n.variance.size + n.gamma.size+ n.tau.size + oversize
            n.ndim = n.mean.ndim
            n.shape = n.mean.shape
            n.num_data = n.mean.shape[0]
            n.input_dim = n.mean.shape[1] if n.ndim != 1 else 1
            return n
        else:
            return super(IBPPosterior, self).__getitem__(s)

class SLVMPrior(VariationalPrior):
    def __init__(self, input_dim, alpha =1., beta=1., Z=None, name='SLVMPrior', **kw):
        super(SLVMPrior, self).__init__(name=name, **kw)
        self.input_dim = input_dim
        self.variance = 1.
        self.alpha = alpha
        self.beta = beta
        self.Z = Z
        if Z is not None:
            assert np.all(np.unique(Z)==np.array([0,1]))

    def KL_divergence(self, variational_posterior):
        mu, S, gamma, tau = variational_posterior.mean.values, variational_posterior.variance.values, variational_posterior.gamma.values, variational_posterior.tau.values
        
        var_mean = np.square(mu)/self.variance
        var_S = (S/self.variance - np.log(S))
        part1 = (gamma* (np.log(self.variance)-1. +var_mean + var_S)).sum()/2.
        
        from scipy.special import betaln,digamma            
        part2 = (gamma*np.log(gamma)).sum() + ((1.-gamma)*np.log(1.-gamma)).sum() + betaln(self.alpha,self.beta)*self.input_dim \
                -betaln(tau[:,0], tau[:,1]).sum() + ((tau[:,0]-(gamma*self.Z).sum(0)-self.alpha)*digamma(tau[:,0])).sum() + \
                ((tau[:,1]-((1-gamma)*self.Z).sum(0)-self.beta)*digamma(tau[:,1])).sum() + ((self.Z.sum(0)+self.alpha+self.beta-tau[:,0]-tau[:,1])*digamma(tau.sum(axis=1))).sum()        

        return part1+part2

    def update_gradients_KL(self, variational_posterior):
        mu, S, gamma, tau = variational_posterior.mean.values, variational_posterior.variance.values, variational_posterior.gamma.values, variational_posterior.tau.values

        variational_posterior.mean.gradient -= gamma*mu/self.variance
        variational_posterior.variance.gradient -= (1./self.variance - 1./S) * gamma /2.
        from scipy.special import digamma,polygamma
        dgamma = np.log(gamma/(1.-gamma))+ (digamma(tau[:,1])-digamma(tau[:,0]))*self.Z
        variational_posterior.binary_prob.gradient -= dgamma+((np.square(mu)+S)/self.variance-np.log(S)+np.log(self.variance)-1.)/2.
        common = (self.Z.sum(0)+self.alpha+self.beta-tau[:,0]-tau[:,1])*polygamma(1,tau.sum(axis=1))
        variational_posterior.tau.gradient[:,0] = -((tau[:,0]-(gamma*self.Z).sum(0)-self.alpha)*polygamma(1,tau[:,0])+common)
        variational_posterior.tau.gradient[:,1] = -((tau[:,1]-((1-gamma)*self.Z).sum(0)-self.beta)*polygamma(1,tau[:,1])+common)


class SSGPLVM(SparseGP_MPI):
    """
    Spike-and-Slab Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, X_variance=None, Gamma=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='Spike_and_Slab GPLVM', group_spike=False, IBP=False,SLVM=False, alpha=2., beta=2., connM=None, tau=None, mpi_comm=None, pi=None, learnPi=False,normalizer=False, sharedX=False, variational_prior=None,**kwargs):

        self.group_spike = group_spike
        self.init = init
        self.sharedX = sharedX
        
        if X is None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        if X_variance is None: # The variance of the variational approximation (S)
            X_variance = np.random.uniform(0,.1,X.shape)
            
        if Gamma is None:
            gamma = np.empty_like(X) # The posterior probabilities of the binary variable in the variational approximation
            gamma[:] = 0.5 + 0.1 * np.random.randn(X.shape[0], input_dim)
            gamma[gamma>1.-1e-9] = 1.-1e-9
            gamma[gamma<1e-9] = 1e-9
        else:
            gamma = Gamma.copy()
                
        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]
        
        if likelihood is None:
            likelihood = Gaussian()

        if kernel is None:
            kernel = kern.RBF(input_dim, lengthscale=fracs, ARD=True) # + kern.white(input_dim)
        if kernel.useGPU:
            kernel.psicomp = PSICOMP_SSRBF_GPU()
        
        if inference_method is None:
            inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)

        if pi is None:
            pi = np.empty((input_dim))
            pi[:] = 0.5
            
        if IBP:
            self.variational_prior = IBPPrior(input_dim=input_dim, alpha=alpha) if variational_prior is None else variational_prior
            X = IBPPosterior(X, X_variance, gamma, tau=tau,sharedX=sharedX)
        elif SLVM:
            self.variational_prior = SLVMPrior(input_dim=input_dim, alpha=alpha, beta=beta, Z=connM) if variational_prior is None else variational_prior
            X = SLVMPosterior(X, X_variance, gamma, tau=tau)
        else:
            self.variational_prior = SpikeAndSlabPrior(pi=pi,learnPi=learnPi, group_spike=group_spike)  if variational_prior is None else variational_prior
            X = SpikeAndSlabPosterior(X, X_variance, gamma, group_spike=group_spike,sharedX=sharedX)

        super(SSGPLVM,self).__init__(X, Y, Z, kernel, likelihood, variational_prior=self.variational_prior, inference_method=inference_method, name=name, mpi_comm=mpi_comm, normalizer=normalizer, **kwargs)
        self.link_parameter(self.X, index=0)
        
    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient, X.binary_prob.gradient = X_grad
    
    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient, X.binary_prob.gradient

    def _propogate_X_val(self):
        pass

    def parameters_changed(self):
        self.X.propogate_val()
        if self.sharedX: self._highest_parent_._propogate_X_val()
        super(SSGPLVM,self).parameters_changed()
        if isinstance(self.inference_method, VarDTC_minibatch):
            self.X.collate_gradient()
            return
        
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient, self.X.binary_prob.gradient = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.grad_dict['dL_dpsi0'], dL_dpsi1=self.grad_dict['dL_dpsi1'], dL_dpsi2=self.grad_dict['dL_dpsi2'])

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.X)
        self.X.collate_gradient()

    def input_sensitivity(self):
        if self.kern.ARD:
            return self.kern.input_sensitivity()
        else:
            return self.variational_prior.pi
        
    def sample_W(self, nSamples, raw_samples=False):
        """
        Sample the loading matrix if the kernel is linear.
        """
        assert isinstance(self.kern, kern.Linear)
        from ..util.linalg import pdinv
        N, D = self.Y.shape
        Q = self.X.shape[1]
        noise_var = self.likelihood.variance.values
        
        # Draw samples for X
        Xs = np.random.randn(*((nSamples,)+self.X.shape))*np.sqrt(self.X.variance.values)+self.X.mean.values
        b = np.random.rand(*((nSamples,)+self.X.shape))
        Xs[b>self.X.gamma.values] = 0
        
        invcov = (Xs[:,:,:,None]*Xs[:,:,None,:]).sum(1)/noise_var+np.eye(Q)
        cov = np.array([pdinv(invcov[s_idx])[0] for s_idx in range(invcov.shape[0])])
        Ws = np.empty((nSamples, Q, D))
        tmp = (np.transpose(Xs, (0,2,1)).reshape(nSamples*Q,N).dot(self.Y)).reshape(nSamples,Q,D)
        mean = (cov[:,:,:,None]*tmp[:,None,:,:]).sum(2)/noise_var
        zeros = np.zeros((Q,))
        for s_idx in range(Xs.shape[0]):
            Ws[s_idx] = (np.random.multivariate_normal(mean=zeros,cov=cov[s_idx],size=(D,))).T+mean[s_idx]
        
        if raw_samples:
            return Ws
        else:
            return Ws.mean(0), Ws.std(0)

        
        

        
