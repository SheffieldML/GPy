# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from matplotlib import pyplot

from ..core.sparse_gp import SparseGP
from .. import kern
from ..likelihoods import Gaussian
from ..inference.optimization import SCG
from ..util import linalg
from ..core.parameterization.variational import SpikeAndSlabPrior, SpikeAndSlabPosterior
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch
from ..inference.latent_function_inference.var_dtc_gpu import VarDTC_GPU


class SSGPLVM(SparseGP):
    """
    Spike-and-Slab Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='Spike-and-Slab GPLVM', group_spike=False, mpi_comm=None, **kwargs):

        self.mpi_comm = mpi_comm
            
        if X == None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if X_variance is None: # The variance of the variational approximation (S)
            X_variance = np.random.uniform(0,.1,X.shape)
            
        gamma = np.empty_like(X, order='F') # The posterior probabilities of the binary variable in the variational approximation
        #gamma[:] = 0.5 + 0.01 * np.random.randn(X.shape[0], input_dim)
        gamma[:] = 0.5
        
        if group_spike:
            gamma[:] = gamma.mean(axis=0)
        
        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        pi = np.empty((input_dim))
        pi[:] = 0.5
        
        if mpi_comm != None:
            mpi_comm.Bcast(X, root=0)
            mpi_comm.Bcast(fracs, root=0)
            mpi_comm.Bcast(X_variance, root=0)
            mpi_comm.Bcast(gamma, root=0)
            mpi_comm.Bcast(Z, root=0)
            mpi_comm.Bcast(pi, root=0)

        if likelihood is None:
            likelihood = Gaussian()

        if kernel is None:
            kernel = kern.RBF(input_dim, lengthscale=fracs, ARD=True) # + kern.white(input_dim)

        self.variational_prior = SpikeAndSlabPrior(pi=pi) # the prior probability of the latent binary variable b
        
        X = SpikeAndSlabPosterior(X, X_variance, gamma)
        
        if group_spike:
            kernel.group_spike_prob = True
            self.variational_prior.group_spike_prob = True

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method, name, **kwargs)
        self.add_parameter(self.X, index=0)
        self.add_parameter(self.variational_prior)
        
        if mpi_comm != None:
            from ..util.mpi import divide_data
            Y_start, Y_end, Y_list = divide_data(Y.shape[0], mpi_comm)
            self.Y_local = self.Y[Y_start:Y_end]
            self.X_local = self.X[Y_start:Y_end]
            self.Y_range = (Y_start, Y_end)
            self.Y_list = np.array(Y_list)
        
    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient, X.binary_prob.gradient = X_grad
    
    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient, X.binary_prob.gradient

    def parameters_changed(self):
        if isinstance(self.inference_method, VarDTC_GPU) or isinstance(self.inference_method, VarDTC_minibatch):
            update_gradients(self, mpi_comm=self.mpi_comm)
            return
        
        super(SSGPLVM, self).parameters_changed()
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient, self.X.binary_prob.gradient = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.grad_dict['dL_dpsi0'], dL_dpsi1=self.grad_dict['dL_dpsi1'], dL_dpsi2=self.grad_dict['dL_dpsi2'])

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.X)

    def input_sensitivity(self):
        if self.kern.ARD:
            return self.kern.input_sensitivity()
        else:
            return self.variational_prior.pi

    def plot_latent(self, plot_inducing=True, *args, **kwargs):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, plot_inducing=plot_inducing, *args, **kwargs)

