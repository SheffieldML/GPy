# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .. import kern
from ..core.sparse_gp_mpi import SparseGP_MPI
from ..likelihoods import Gaussian
from ..core.parameterization.variational import NormalPosterior, GmmNormalPrior
from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
import logging

class GmmBayesianGPLVM(SparseGP_MPI):
    """
    Gaussian mixture model Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data (np.ndarray) or GPy.likelihood
    :type Y: np.ndarray| GPy.likelihood instance
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, X=None, X_variance=None, init='PCA', n_component=2, num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None,
                 name='gmm bayesian gplvm', mpi_comm=None, normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1, Y_metadata=None):

        self.logger = logging.getLogger(self.__class__.__name__)
        if X is None:
            from ..util.initialization import initialize_latent
            self.logger.info("initializing latent space X with method {}".format(init))
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if X_variance is None:
            self.logger.info("initializing latent space variance ~ uniform(0,.1)")
            X_variance = np.random.uniform(0,.1,X.shape)

        if Z is None:
            self.logger.info("initializing inducing inputs")
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            self.logger.info("initializing kernel RBF")
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) #+ kern.Bias(input_dim) + kern.White(input_dim)

        if likelihood is None:
            likelihood = Gaussian()

        # Need to define what the model is initialised like
        pi = np.ones(n_component) / float(n_component) # p(k)
        variational_pi = pi.copy()
        # px_mu = np.zeros(n_component)
        # px_var = np.ones(n_component)
        px_mu = [[]] * n_component
        px_var = [[]] * n_component
        for i in range(n_component):
            px_mu[i] = np.zeros_like(X_variance)
            px_var[i] = np.ones_like(X_variance)
        

        # print("Should print")
        # print(pi)
        # print(px_mu)
        # print(px_var)
        # print(variational_pi)
        # print("Didnt print")
        self.variational_prior = GmmNormalPrior(px_mu=px_mu, px_var=px_var, pi=pi,
                                n_component=n_component, variational_pi=variational_pi)

        X = NormalPosterior(X, X_variance)

        if inference_method is None:
            if mpi_comm is not None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                from ..inference.latent_function_inference.var_dtc import VarDTC
                self.logger.debug("creating inference_method var_dtc")
                inference_method = VarDTC(limit=1 if not missing_data else Y.shape[1])
        if isinstance(inference_method,VarDTC_minibatch):
            inference_method.mpi_comm = mpi_comm

        super(GmmBayesianGPLVM,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method,
                                           normalizer=normalizer, mpi_comm=mpi_comm,
                                           variational_prior=self.variational_prior,
                                           Y_metadata=Y_metadata
                                           )
        self.link_parameter(self.X, index=0)

    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient = X_grad

    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient

    def parameters_changed(self):
        super(GmmBayesianGPLVM,self).parameters_changed()
        if isinstance(self.inference_method, VarDTC_minibatch):
            return

        kl_fctr = 1.
        self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z,
                                            dL_dpsi0=self.grad_dict['dL_dpsi0'],
                                            dL_dpsi1=self.grad_dict['dL_dpsi1'],
                                            dL_dpsi2=self.grad_dict['dL_dpsi2'])

        self.variational_prior.update_gradients_KL(self.X)


        #super(BayesianGPLVM, self).parameters_changed()
        #self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

        #self.X.mean.gradient, self.X.variance.gradient = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, dL_dpsi0=self.grad_dict['dL_dpsi0'], dL_dpsi1=self.grad_dict['dL_dpsi1'], dL_dpsi2=self.grad_dict['dL_dpsi2'])

        # This is testing code -------------------------
#         i = np.random.randint(self.X.shape[0])
#         X_ = self.X.mean
#         which = np.sqrt(((X_ - X_[i:i+1])**2).sum(1)).argsort()>(max(0, self.X.shape[0]-51))
#         _, _, grad_dict = self.inference_method.inference(self.kern, self.X[which], self.Z, self.likelihood, self.Y[which], self.Y_metadata)
#         grad = self.kern.gradients_qX_expectations(variational_posterior=self.X[which], Z=self.Z, dL_dpsi0=grad_dict['dL_dpsi0'], dL_dpsi1=grad_dict['dL_dpsi1'], dL_dpsi2=grad_dict['dL_dpsi2'])
#
#         self.X.mean.gradient[:] = 0
#         self.X.variance.gradient[:] = 0
#         self.X.mean.gradient[which] = grad[0]
#         self.X.variance.gradient[which] = grad[1]

        # update for the KL divergence
#         self.variational_prior.update_gradients_KL(self.X, which)
        # -----------------------------------------------

        # update for the KL divergence
        #self.variational_prior.update_gradients_KL(self.X)

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
        N_test = Y.shape[0]
        input_dim = self.Z.shape[1]

        means = np.zeros((N_test, input_dim))
        covars = np.zeros((N_test, input_dim))

        dpsi0 = -0.5 * self.input_dim / self.likelihood.variance
        dpsi2 = self.grad_dict['dL_dpsi2'][0][None, :, :] # TODO: this may change if we ignore het. likelihoods
        V = Y/self.likelihood.variance

        #compute CPsi1V
        #if self.Cpsi1V is None:
        #    psi1V = np.dot(self.psi1.T, self.likelihood.V)
        #    tmp, _ = linalg.dtrtrs(self._Lm, np.asfortranarray(psi1V), lower=1, trans=0)
        #    tmp, _ = linalg.dpotrs(self.LB, tmp, lower=1)
        #    self.Cpsi1V, _ = linalg.dtrtrs(self._Lm, tmp, lower=1, trans=1)

        dpsi1 = np.dot(self.posterior.woodbury_vector, V.T)

        #start = np.zeros(self.input_dim * 2)


        from scipy.optimize import minimize

        for n, dpsi1_n in enumerate(dpsi1.T[:, :, None]):
            args = (input_dim, self.kern.copy(), self.Z, dpsi0, dpsi1_n.T, dpsi2)
            res = minimize(latent_cost_and_grad, jac=True, x0=np.hstack((means[n], covars[n])), args=args, method='BFGS')
            xopt = res.x
            mu, log_S = xopt.reshape(2, 1, -1)
            means[n] = mu[0].copy()
            covars[n] = np.exp(log_S[0]).copy()

        X = NormalPosterior(means, covars)

        return X

    def dmu_dX(self, Xnew):
        """
        Calculate the gradient of the prediction at Xnew w.r.t Xnew.
        """
        dmu_dX = np.zeros_like(Xnew)
        for i in range(self.Z.shape[0]):
            dmu_dX += self.kern.gradients_X(self.grad_dict['dL_dpsi1'][i:i + 1, :], Xnew, self.Z[i:i + 1, :])
        return dmu_dX

    def dmu_dXnew(self, Xnew):
        """
        Individual gradient of prediction at Xnew w.r.t. each sample in Xnew
        """
        gradients_X = np.zeros((Xnew.shape[0], self.num_inducing))
        ones = np.ones((1, 1))
        for i in range(self.Z.shape[0]):
            gradients_X[:, i] = self.kern.gradients_X(ones, Xnew, self.Z[i:i + 1, :]).sum(-1)
        return np.dot(gradients_X, self.grad_dict['dL_dpsi1'])

    def plot_steepest_gradient_map(self, *args, ** kwargs):
        """
        See GPy.plotting.matplot_dep.dim_reduction_plots.plot_steepest_gradient_map
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_steepest_gradient_map(self,*args,**kwargs)


def latent_cost_and_grad(mu_S, input_dim, kern, Z, dL_dpsi0, dL_dpsi1, dL_dpsi2):
    """
    objective function for fitting the latent variables for test points
    (negative log-likelihood: should be minimised!)
    """
    mu = mu_S[:input_dim][None]
    log_S = mu_S[input_dim:][None]
    S = np.exp(log_S)

    X = NormalPosterior(mu, S)

    psi0 = kern.psi0(Z, X)
    psi1 = kern.psi1(Z, X)
    psi2 = kern.psi2(Z, X)

    lik = dL_dpsi0 * psi0.sum() + np.einsum('ij,kj->...', dL_dpsi1, psi1) + np.einsum('ijk,lkj->...', dL_dpsi2, psi2) - 0.5 * np.sum(np.square(mu) + S) + 0.5 * np.sum(log_S)

    dLdmu, dLdS = kern.gradients_qX_expectations(dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, X)
    dmu = dLdmu - mu
    # dS = S0 + S1 + S2 -0.5 + .5/S
    dlnS = S * (dLdS - 0.5) + .5

    return -lik, -np.hstack((dmu.flatten(), dlnS.flatten()))
