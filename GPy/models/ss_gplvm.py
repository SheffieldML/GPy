# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from ..core.sparse_gp import SparseGP
from .. import kern
from ..likelihoods import Gaussian
from ..core.parameterization.variational import SpikeAndSlabPrior, SpikeAndSlabPosterior
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch
from ..inference.latent_function_inference.var_dtc_gpu import VarDTC_GPU
from ..kern._src.psi_comp.ssrbf_psi_gpucomp import PSICOMP_SSRBF_GPU

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
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='Spike_and_Slab GPLVM', group_spike=False, mpi_comm=None, pi=None, learnPi=True, **kwargs):

        self.mpi_comm = mpi_comm
        self.__IN_OPTIMIZATION__ = False
        self.group_spike = group_spike
        
        if X == None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if X_variance is None: # The variance of the variational approximation (S)
            X_variance = np.random.uniform(0,.1,X.shape)
            
        gamma = np.empty_like(X) # The posterior probabilities of the binary variable in the variational approximation
        gamma[:] = 0.5 + 0.1 * np.random.randn(X.shape[0], input_dim)
        gamma[gamma>1.-1e-9] = 1.-1e-9
        gamma[gamma<1e-9] = 1e-9
                
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
        self.variational_prior = SpikeAndSlabPrior(pi=pi,learnPi=learnPi) # the prior probability of the latent binary variable b
        
        X = SpikeAndSlabPosterior(X, X_variance, gamma)
        
        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method, name, **kwargs)
        self.add_parameter(self.X, index=0)
        self.add_parameter(self.variational_prior)
                
        if mpi_comm != None:
            from ..util.mpi import divide_data
            N_start, N_end, N_list = divide_data(Y.shape[0], mpi_comm)
            self.N_range = (N_start, N_end)
            self.N_list = np.array(N_list)
            self.Y_local = self.Y[N_start:N_end]
            print 'MPI RANK: '+str(self.mpi_comm.rank)+' with datasize: '+str(self.N_range)
            mpi_comm.Bcast(self.param_array, root=0)
        
        if self.group_spike:
            [self.X.gamma[:,i].tie('tieGamma'+str(i)) for i in xrange(self.X.gamma.shape[1])] # Tie columns together
        
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

    def __getstate__(self):
        dc = super(SSGPLVM, self).__getstate__()
        dc['mpi_comm'] = None
        if self.mpi_comm != None:
            del dc['N_range']
            del dc['N_list']
            del dc['Y_local']
        return dc
 
    def __setstate__(self, state):
        return super(SSGPLVM, self).__setstate__(state)
    
    #=====================================================
    # The MPI parallelization 
    #     - can move to model at some point
    #=====================================================
    
    def _set_params_transformed(self, p):
        if self.mpi_comm != None:
            if self.__IN_OPTIMIZATION__ and self.mpi_comm.rank==0:
                self.mpi_comm.Bcast(np.int32(1),root=0)
            self.mpi_comm.Bcast(p, root=0)
        super(SSGPLVM, self)._set_params_transformed(p)
        
    def optimize(self, optimizer=None, start=None, **kwargs):
        self.__IN_OPTIMIZATION__ = True
        if self.mpi_comm==None:
            super(SSGPLVM, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==0:
            super(SSGPLVM, self).optimize(optimizer,start,**kwargs)
            self.mpi_comm.Bcast(np.int32(-1),root=0)
        elif self.mpi_comm.rank>0:
            x = self._get_params_transformed().copy()
            flag = np.empty(1,dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag,root=0)
                if flag==1:
                    self._set_params_transformed(x)
                elif flag==-1:
                    break
                else:
                    self.__IN_OPTIMIZATION__ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self.__IN_OPTIMIZATION__ = False
