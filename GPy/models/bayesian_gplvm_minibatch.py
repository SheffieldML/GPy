# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .. import kern
from ..likelihoods import Gaussian
from ..core.parameterization.variational import NormalPosterior, NormalPrior
from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
import logging
from GPy.models.sparse_gp_minibatch import SparseGPMiniBatch
from GPy.core.parameterization.param import Param
from GPy.core.parameterization.observable_array import ObsAr

class BayesianGPLVMMiniBatch(SparseGPMiniBatch):
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
                 Z=None, kernel=None, inference_method=None, likelihood=None,
                 name='bayesian gplvm', normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1):
        self.logger = logging.getLogger(self.__class__.__name__)
        if X is None:
            from ..util.initialization import initialize_latent
            self.logger.info("initializing latent space X with method {}".format(init))
            X, fracs = initialize_latent(init, input_dim, Y)
        else:
            fracs = np.ones(input_dim)

        self.init = init

        if Z is None:
            self.logger.info("initializing inducing inputs")
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if X_variance == False:
            self.logger.info('no variance on X, activating sparse GPLVM')
            X = Param("latent space", X)
        elif X_variance is None:
            self.logger.info("initializing latent space variance ~ uniform(0,.1)")
            X_variance = np.random.uniform(0,.1,X.shape)
            self.variational_prior = NormalPrior()
            X = NormalPosterior(X, X_variance)

        if kernel is None:
            self.logger.info("initializing kernel RBF")
            kernel = kern.RBF(input_dim, lengthscale=1./fracs, ARD=True) #+ kern.Bias(input_dim) + kern.White(input_dim)

        if likelihood is None:
            likelihood = Gaussian()

        self.kl_factr = 1.

        if inference_method is None:
            from ..inference.latent_function_inference.var_dtc import VarDTC
            self.logger.debug("creating inference_method var_dtc")
            inference_method = VarDTC(limit=1 if not missing_data else Y.shape[1])

        super(BayesianGPLVMMiniBatch,self).__init__(X, Y, Z, kernel, likelihood=likelihood,
                                           name=name, inference_method=inference_method,
                                           normalizer=normalizer,
                                           missing_data=missing_data, stochastic=stochastic,
                                           batchsize=batchsize)
        self.X = X
        self.link_parameter(self.X, 0)

    def set_X_gradients(self, X, X_grad):
        """Set the gradients of the posterior distribution of X in its specific form."""
        X.mean.gradient, X.variance.gradient = X_grad

    def get_X_gradients(self, X):
        """Get the gradients of the posterior distribution of X in its specific form."""
        return X.mean.gradient, X.variance.gradient

    def _inner_parameters_changed(self, kern, X, Z, likelihood, Y, Y_metadata, Lm=None, dL_dKmm=None, psi0=None, psi1=None, psi2=None, **kw):
        posterior, log_marginal_likelihood, grad_dict = super(BayesianGPLVMMiniBatch, self)._inner_parameters_changed(kern, X, Z, likelihood, Y, Y_metadata, Lm=Lm, dL_dKmm=dL_dKmm,
                                                                                                                    psi0=psi0, psi1=psi1, psi2=psi2, **kw)
        return posterior, log_marginal_likelihood, grad_dict

    def _outer_values_update(self, full_values):
        """
        Here you put the values, which were collected before in the right places.
        E.g. set the gradients of parameters, etc.
        """
        super(BayesianGPLVMMiniBatch, self)._outer_values_update(full_values)
        if self.has_uncertain_inputs():
            meangrad_tmp, vargrad_tmp = self.kern.gradients_qX_expectations(
                                            variational_posterior=self.X,
                                            Z=self.Z, dL_dpsi0=full_values['dL_dpsi0'],
                                            dL_dpsi1=full_values['dL_dpsi1'],
                                            dL_dpsi2=full_values['dL_dpsi2'],
                                            psi0=self.psi0, psi1=self.psi1, psi2=self.psi2)

            self.X.mean.gradient = meangrad_tmp
            self.X.variance.gradient = vargrad_tmp
        else:
            self.X.gradient = self.kern.gradients_X(full_values['dL_dKnm'], self.X, self.Z)
            self.X.gradient += self.kern.gradients_X_diag(full_values['dL_dKdiag'], self.X)

    def _outer_init_full_values(self):
        return super(BayesianGPLVMMiniBatch, self)._outer_init_full_values()

    def parameters_changed(self):
        super(BayesianGPLVMMiniBatch,self).parameters_changed()

        kl_fctr = self.kl_factr
        if kl_fctr > 0:
            Xgrad = self.X.gradient.copy()
            self.X.gradient[:] = 0
            self.variational_prior.update_gradients_KL(self.X)

            if self.missing_data or not self.stochastics:
                self.X.mean.gradient = kl_fctr*self.X.mean.gradient
                self.X.variance.gradient = kl_fctr*self.X.variance.gradient
            else:
                d = self.output_dim
                self.X.mean.gradient = kl_fctr*self.X.mean.gradient*self.stochastics.batchsize/d
                self.X.variance.gradient = kl_fctr*self.X.variance.gradient*self.stochastics.batchsize/d
            self.X.gradient += Xgrad

            if self.missing_data or not self.stochastics:
                self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)
            elif self.stochastics:
                d = self.output_dim
                self._log_marginal_likelihood -= kl_fctr*self.variational_prior.KL_divergence(self.X)*self.stochastics.batchsize/d

        self._Xgrad = self.X.gradient.copy()

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
