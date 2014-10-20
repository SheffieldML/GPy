# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import kern
from ..likelihoods import Gaussian
from ..core.parameterization.variational import SpikeAndSlabPrior, SpikeAndSlabPosterior
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch
from ..inference.latent_function_inference.var_dtc_gpu import VarDTC_GPU
from ..kern._src.psi_comp.ssrbf_psi_gpucomp import PSICOMP_SSRBF_GPU

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
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='Spike_and_Slab GPLVM', 
                 group_spike=False, mpi_comm=None, pi=None, learnPi=True,normalizer=False, variational_prior=None,**kwargs):

        self.group_spike = group_spike
        self.init = init
        
        X, fracs = self._init_X(input_dim, Y, X, X_variance, Gamma, init)
                
        if Z is None:
            Z = np.random.permutation(X.mean.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]
        
        if likelihood is None:
            likelihood = Gaussian()

        if kernel is None:
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) # + kern.white(input_dim)
        if kernel.useGPU:
            kernel.psicomp = PSICOMP_SSRBF_GPU()
        
        if inference_method is None:
            inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)

        if pi is None:
            pi = np.empty((input_dim))
            pi[:] = 0.5

        if variational_prior is None:
            self.variational_prior = SpikeAndSlabPrior(pi=pi,learnPi=learnPi, group_spike=group_spike) # the prior probability of the latent binary variable b
        else:
            self.variational_prior = variational_prior
        
        super(SSGPLVM,self).__init__(X, Y, Z, kernel, likelihood, variational_prior=self.variational_prior, inference_method=inference_method, name=name, mpi_comm=mpi_comm, normalizer=normalizer, **kwargs)
#         self.X.unfix()
#         self.X.variance.constrain_positive()
                
        if self.group_spike:
            [self.X.gamma[:,i].tie_together() for i in xrange(self.X.gamma.shape[1])] # Tie columns together
            
    def _init_X(self, input_dim, Y=None, X=None, X_variance=None, Gamma=None, init='PCA'):
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
        
        return SpikeAndSlabPosterior(X, X_variance, gamma), fracs

    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient, X.binary_prob.gradient = X_grad
    
    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient, X.binary_prob.gradient

    def parameters_changed(self):
        super(SSGPLVM,self).parameters_changed()
        if isinstance(self.inference_method, VarDTC_minibatch):
            return
        
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient, self.X.binary_prob.gradient = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.grad_dict['dL_dpsi0'], dL_dpsi1=self.grad_dict['dL_dpsi1'], dL_dpsi2=self.grad_dict['dL_dpsi2'])

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.X)

    def input_sensitivity(self):
        if self.kern.ARD:
            return self.kern.input_sensitivity()
        else:
            return self.variational_prior.pi

    def plot_latent(self, plot_inducing=True, interactive=False, *args, **kwargs):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        if interactive:
            return dim_reduction_plots.plot_latent_interactive(self, **kwargs)
        else:
            return dim_reduction_plots.plot_latent(self, plot_inducing=plot_inducing, *args, **kwargs)
        
    def inference_X(self, Y_new, optimize=True):
        from ..inference.latent_function_inference.inference_X import inference_newX
        return inference_newX(self, Y_new, optimize=optimize)


