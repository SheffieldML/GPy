# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from gplvm import GPLVM
from .. import kern
from ..core import SparseGP
from ..likelihoods import Gaussian
from ..inference.optimization import SCG
from ..util import linalg
from ..core.parameterization.variational import Normal

class BayesianGPLVM(SparseGP, GPLVM):
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
                 Z=None, kernel=None, inference_method=None, likelihood=Gaussian(), name='bayesian gplvm', **kwargs):
        if X == None:
            X = self.initialise_latent(init, input_dim, Y)
        self.init = init

        if X_variance is None:
            X_variance = np.clip((np.ones_like(X) * 0.5) + .01 * np.random.randn(*X.shape), 0.001, 1)

        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.rbf(input_dim) # + kern.white(input_dim)

        self.q = Normal(X, X_variance)
        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method, X_variance, name, **kwargs)
        self.add_parameter(self.q, index=0)
        self.ensure_default_constraints()

    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return SparseGP._getstate(self) + [self.init]

    def _setstate(self, state):
        self._const_jitter = None
        self.init = state.pop()
        SparseGP._setstate(self, state)

#     def _get_param_names(self):
#         X_names = sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
#         S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
#         return (X_names + S_names + SparseGP._get_param_names(self))

    #def _get_print_names(self):
    #    return SparseGP._get_print_names(self)

#     def _get_params(self):
#         """
#         Horizontally stacks the parameters in order to present them to the optimizer.
#         The resulting 1-input_dim array has this structure:
#
#         ===============================================================
#         |       mu       |        S        |    Z    | theta |  beta  |
#         ===============================================================
#
#         """
#         x = np.hstack((self.X.flatten(), self.X_variance.flatten(), SparseGP._get_params(self)))
#         return x

#     def _set_params(self, x, save_old=True, save_count=0):
#         N, input_dim = self.num_data, self.input_dim
#         self.X = x[:self.X.size].reshape(N, input_dim).copy()
#         self.X_variance = x[(N * input_dim):(2 * N * input_dim)].reshape(N, input_dim).copy()
#         SparseGP._set_params(self, x[(2 * N * input_dim):])

    def dKL_dmuS(self):
        dKL_dS = (1. - (1. / (self.X_variance))) * 0.5
        dKL_dmu = self.X
        return dKL_dmu, dKL_dS

    def dL_dmuS(self):
        dL_dmu_psi0, dL_dS_psi0 = self.kern.dpsi0_dmuS(self.grad_dict['dL_dpsi0'], self.Z, self.X, self.X_variance)
        dL_dmu_psi1, dL_dS_psi1 = self.kern.dpsi1_dmuS(self.grad_dict['dL_dpsi1'], self.Z, self.X, self.X_variance)
        dL_dmu_psi2, dL_dS_psi2 = self.kern.dpsi2_dmuS(self.grad_dict['dL_dpsi2'], self.Z, self.X, self.X_variance)
        dL_dmu = dL_dmu_psi0 + dL_dmu_psi1 + dL_dmu_psi2
        dL_dS = dL_dS_psi0 + dL_dS_psi1 + dL_dS_psi2

        return dL_dmu, dL_dS

    def KL_divergence(self):
        var_mean = np.square(self.X).sum()
        var_S = np.sum(self.X_variance - np.log(self.X_variance))
        return 0.5 * (var_mean + var_S) - 0.5 * self.input_dim * self.num_data

    def parameters_changed(self):
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.X_variance, self.Z, self.likelihood, self.Y)
        self._log_marginal_likelihood -= self.KL_divergence()

        #The derivative of the bound wrt the inducing inputs Z
        self.Z.gradient = self.kern.gradients_X(self.grad_dict['dL_dKmm'], self.Z)
        self.Z.gradient += self.kern.dpsi1_dZ(self.grad_dict['dL_dpsi1'], self.Z, self.X, self.X_variance)
        self.Z.gradient += self.kern.dpsi2_dZ(self.grad_dict['dL_dpsi2'], self.Z, self.X, self.X_variance)
        
        dL_dmu, dL_dS = self.dL_dmuS()
        dKL_dmu, dKL_dS = self.dKL_dmuS()
        self.q.means.gradient = dL_dmu - dKL_dmu
        self.q.variances.gradient = dL_dS - dKL_dS
    

#     def log_likelihood(self):        
#         ll = SparseGP.log_likelihood(self)
#         kl = self.KL_divergence()
#         return ll - kl

    def _dbound_dmuS(self):
        dKL_dmu, dKL_dS = self.dKL_dmuS()
        dL_dmu, dL_dS = self.dL_dmuS()
        d_dmu = (dL_dmu - dKL_dmu).flatten()
        d_dS = (dL_dS - dKL_dS).flatten()
        return np.hstack((d_dmu, d_dS))

#     def _log_likelihood_gradients(self):
#         dKL_dmu, dKL_dS = self.dKL_dmuS()
#         dL_dmu, dL_dS = self.dL_dmuS()
#         d_dmu = (dL_dmu - dKL_dmu).flatten()
#         d_dS = (dL_dS - dKL_dS).flatten()
#         self.dbound_dmuS = np.hstack((d_dmu, d_dS))
#         self.dbound_dZtheta = SparseGP._log_likelihood_gradients(self)
#         return np.hstack((self.dbound_dmuS.flatten(), self.dbound_dZtheta))

    def plot_latent(self, plot_inducing=True, *args, **kwargs):
        """
        See GPy.plotting.matplot_dep.dim_reduction_plots.plot_latent
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_latent(self, plot_inducing=plot_inducing, *args, **kwargs)

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
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots

        return dim_reduction_plots.plot_steepest_gradient_map(model,*args,**kwargs)

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
