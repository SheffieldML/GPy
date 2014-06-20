# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .. import kern
from ..core import SparseGP
from ..likelihoods import Gaussian
from ..inference.optimization import SCG
from ..util import linalg
from ..core.parameterization.variational import NormalPosterior, NormalPrior, VariationalPosterior
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch
from ..inference.latent_function_inference.var_dtc_gpu import VarDTC_GPU

class BayesianGPLVM(SparseGP):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='bayesian gplvm', mpi_comm=None, **kwargs):
        self.mpi_comm = mpi_comm
        self.__IN_OPTIMIZATION__ = False
        if X == None:
            from ..util.initialization import initialize_latent
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if X_variance is None:
            X_variance = np.random.uniform(0,.1,X.shape)


        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) # + kern.white(input_dim)

        if likelihood is None:
            likelihood = Gaussian()


        self.variational_prior = NormalPrior()
        X = NormalPosterior(X, X_variance)

        if inference_method is None:
            if np.any(np.isnan(Y)):
                from ..inference.latent_function_inference.var_dtc import VarDTCMissingData
                inference_method = VarDTCMissingData()
            elif mpi_comm != None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                from ..inference.latent_function_inference.var_dtc import VarDTC
                inference_method = VarDTC()
                
        if kernel.useGPU and isinstance(inference_method, VarDTC_GPU):
            kernel.psicomp.GPU_direct = True

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method, name, **kwargs)
        self.add_parameter(self.X, index=0)

        if mpi_comm != None:
            from ..util.mpi import divide_data
            Y_start, Y_end, Y_list = divide_data(Y.shape[0], mpi_comm)
            self.Y_local = self.Y[Y_start:Y_end]
            self.X_local = self.X[Y_start:Y_end]
            self.Y_range = (Y_start, Y_end)
            self.Y_list = np.array(Y_list)
            mpi_comm.Bcast(self.param_array, root=0)

    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient = X_grad
    
    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient

    def parameters_changed(self):
        if isinstance(self.inference_method, VarDTC_GPU) or isinstance(self.inference_method, VarDTC_minibatch):
            update_gradients(self, mpi_comm=self.mpi_comm)
            return

        super(BayesianGPLVM, self).parameters_changed()
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.grad_dict['dL_dpsi0'], dL_dpsi1=self.grad_dict['dL_dpsi1'], dL_dpsi2=self.grad_dict['dL_dpsi2'])

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.X)

    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None, 
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, labels, which_indices,
                resolution, ax, marker, s,
                fignum, plot_inducing, legend,
                plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)

    def do_test_latents(self, Y):
        """
        Compute the latent representation for a set of new points Y

        Notes:
        This will only work with a univariate Gaussian likelihood (for now)
        """
        assert not self.likelihood.is_heteroscedastic
        N_test = Y.shape[0]
        input_dim = self.Z.shape[1]
        means = np.zeros((N_test, input_dim))
        covars = np.zeros((N_test, input_dim))

        dpsi0 = -0.5 * self.input_dim * self.likelihood.precision
        dpsi2 = self.dL_dpsi2[0][None, :, :] # TODO: this may change if we ignore het. likelihoods
        V = self.likelihood.precision * Y

        #compute CPsi1V
        if self.Cpsi1V is None:
            psi1V = np.dot(self.psi1.T, self.likelihood.V)
            tmp, _ = linalg.dtrtrs(self._Lm, np.asfortranarray(psi1V), lower=1, trans=0)
            tmp, _ = linalg.dpotrs(self.LB, tmp, lower=1)
            self.Cpsi1V, _ = linalg.dtrtrs(self._Lm, tmp, lower=1, trans=1)

        dpsi1 = np.dot(self.Cpsi1V, V.T)

        start = np.zeros(self.input_dim * 2)

        for n, dpsi1_n in enumerate(dpsi1.T[:, :, None]):
            args = (self.kern, self.Z, dpsi0, dpsi1_n.T, dpsi2)
            xopt, fopt, neval, status = SCG(f=latent_cost, gradf=latent_grad, x=start, optargs=args, display=False)

            mu, log_S = xopt.reshape(2, 1, -1)
            means[n] = mu[0].copy()
            covars[n] = np.exp(log_S[0]).copy()

        return means, covars

    def dmu_dX(self, Xnew):
        """
        Calculate the gradient of the prediction at Xnew w.r.t Xnew.
        """
        dmu_dX = np.zeros_like(Xnew)
        for i in range(self.Z.shape[0]):
            dmu_dX += self.kern.gradients_X(self.Cpsi1Vf[i:i + 1, :], Xnew, self.Z[i:i + 1, :])
        return dmu_dX

    def dmu_dXnew(self, Xnew):
        """
        Individual gradient of prediction at Xnew w.r.t. each sample in Xnew
        """
        gradients_X = np.zeros((Xnew.shape[0], self.num_inducing))
        ones = np.ones((1, 1))
        for i in range(self.Z.shape[0]):
            gradients_X[:, i] = self.kern.gradients_X(ones, Xnew, self.Z[i:i + 1, :]).sum(-1)
        return np.dot(gradients_X, self.Cpsi1Vf)

    def plot_steepest_gradient_map(self, *args, ** kwargs):
        """
        See GPy.plotting.matplot_dep.dim_reduction_plots.plot_steepest_gradient_map
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_steepest_gradient_map(self,*args,**kwargs)

    def __getstate__(self):
        dc = super(BayesianGPLVM, self).__getstate__()
        dc['mpi_comm'] = None
        if self.mpi_comm != None:
            del dc['Y_local']
            del dc['X_local']
            del dc['Y_range']
        return dc
 
    def __setstate__(self, state):
        return super(BayesianGPLVM, self).__setstate__(state)
    
    #=====================================================
    # The MPI parallelization 
    #     - can move to model at some point
    #=====================================================
    
    def _set_params_transformed(self, p):
        if self.mpi_comm != None:
            if self.__IN_OPTIMIZATION__ and self.mpi_comm.rank==0:
                self.mpi_comm.Bcast(np.int32(1),root=0)
            self.mpi_comm.Bcast(p, root=0)
        super(BayesianGPLVM, self)._set_params_transformed(p)
        
    def optimize(self, optimizer=None, start=None, **kwargs):
        self.__IN_OPTIMIZATION__ = True
        if self.mpi_comm==None:
            super(BayesianGPLVM, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==0:
            super(BayesianGPLVM, self).optimize(optimizer,start,**kwargs)
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

def latent_cost_and_grad(mu_S, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    objective function for fitting the latent variables for test points
    (negative log-likelihood: should be minimised!)
    """
    mu, log_S = mu_S.reshape(2, 1, -1)
    S = np.exp(log_S)

    psi0 = kern.psi0(Z, mu, S)
    psi1 = kern.psi1(Z, mu, S)
    psi2 = kern.psi2(Z, mu, S)

    lik = dL_dpsi0 * psi0 + np.dot(dL_dpsi1.flatten(), psi1.flatten()) + np.dot(dL_dpsi2.flatten(), psi2.flatten()) - 0.5 * np.sum(np.square(mu) + S) + 0.5 * np.sum(log_S)

    mu0, S0 = kern.dpsi0_dmuS(dL_dpsi0, Z, mu, S)
    mu1, S1 = kern.dpsi1_dmuS(dL_dpsi1, Z, mu, S)
    mu2, S2 = kern.dpsi2_dmuS(dL_dpsi2, Z, mu, S)

    dmu = mu0 + mu1 + mu2 - mu
    # dS = S0 + S1 + S2 -0.5 + .5/S
    dlnS = S * (S0 + S1 + S2 - 0.5) + .5
    return -lik, -np.hstack((dmu.flatten(), dlnS.flatten()))

def latent_cost(mu_S, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    objective function for fitting the latent variables (negative log-likelihood: should be minimised!)
    This is the same as latent_cost_and_grad but only for the objective
    """
    mu, log_S = mu_S.reshape(2, 1, -1)
    S = np.exp(log_S)

    psi0 = kern.psi0(Z, mu, S)
    psi1 = kern.psi1(Z, mu, S)
    psi2 = kern.psi2(Z, mu, S)

    lik = dL_dpsi0 * psi0 + np.dot(dL_dpsi1.flatten(), psi1.flatten()) + np.dot(dL_dpsi2.flatten(), psi2.flatten()) - 0.5 * np.sum(np.square(mu) + S) + 0.5 * np.sum(log_S)
    return -float(lik)

def latent_grad(mu_S, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    This is the same as latent_cost_and_grad but only for the grad
    """
    mu, log_S = mu_S.reshape(2, 1, -1)
    S = np.exp(log_S)

    mu0, S0 = kern.dpsi0_dmuS(dL_dpsi0, Z, mu, S)
    mu1, S1 = kern.dpsi1_dmuS(dL_dpsi1, Z, mu, S)
    mu2, S2 = kern.dpsi2_dmuS(dL_dpsi2, Z, mu, S)

    dmu = mu0 + mu1 + mu2 - mu
    # dS = S0 + S1 + S2 -0.5 + .5/S
    dlnS = S * (S0 + S1 + S2 - 0.5) + .5

    return -np.hstack((dmu.flatten(), dlnS.flatten()))
