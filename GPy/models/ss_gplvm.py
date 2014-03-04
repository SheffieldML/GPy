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
                 Z=None, kernel=None, inference_method=None, likelihood=None, name='Spike-and-Slab GPLVM', **kwargs):

        if X == None: # The mean of variational approximation (mu)
            from ..util.initialization import initialize_latent
            X = initialize_latent(init, input_dim, Y)
        self.init = init

        if X_variance is None: # The variance of the variational approximation (S)
            X_variance = np.random.uniform(0,.1,X.shape)
            
        gamma = np.empty_like(X) # The posterior probabilities of the binary variable in the variational approximation
        gamma[:] = 0.5 + 0.01 * np.random.randn(X.shape[0], input_dim)
        
        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if likelihood is None:
            likelihood = Gaussian()

        if kernel is None:
            kernel = kern.SSRBF(input_dim)
            
        pi = np.empty((input_dim))
        pi[:] = 0.5
        self.variational_prior = SpikeAndSlabPrior(pi=pi) # the prior probability of the latent binary variable b
        X = SpikeAndSlabPosterior(X, X_variance, gamma)

        SparseGP.__init__(self, X, Y, Z, kernel, likelihood, inference_method, name, **kwargs)
        self.add_parameter(self.X, index=0)
        self.add_parameter(self.variational_prior)

    def parameters_changed(self):
        super(SSGPLVM, self).parameters_changed()
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

        self.X.mean.gradient, self.X.variance.gradient, self.X.binary_prob.gradient = self.kern.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, **self.grad_dict)

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.X)

    def plot_latent(self, plot_inducing=True, *args, **kwargs):
        import sys
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

        dpsi0 = -0.5 * self.output_dim * self.likelihood.precision
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
            dmu_dX += self.kern.dK_dX(self.Cpsi1Vf[i:i + 1, :], Xnew, self.Z[i:i + 1, :])
        return dmu_dX

    def dmu_dXnew(self, Xnew):
        """
        Individual gradient of prediction at Xnew w.r.t. each sample in Xnew
        """
        dK_dX = np.zeros((Xnew.shape[0], self.num_inducing))
        ones = np.ones((1, 1))
        for i in range(self.Z.shape[0]):
            dK_dX[:, i] = self.kern.dK_dX(ones, Xnew, self.Z[i:i + 1, :]).sum(-1)
        return np.dot(dK_dX, self.Cpsi1Vf)

    def plot_steepest_gradient_map(self, fignum=None, ax=None, which_indices=None, labels=None, data_labels=None, data_marker='o', data_s=40, resolution=20, aspect='auto', updates=False, ** kwargs):
        input_1, input_2 = significant_dims = most_significant_input_dimensions(self, which_indices)

        X = np.zeros((resolution ** 2, self.input_dim))
        indices = np.r_[:X.shape[0]]
        if labels is None:
            labels = range(self.output_dim)

        def plot_function(x):
            X[:, significant_dims] = x
            dmu_dX = self.dmu_dXnew(X)
            argmax = np.argmax(dmu_dX, 1)
            return dmu_dX[indices, argmax], np.array(labels)[argmax]

        if ax is None:
            fig = pyplot.figure(num=fignum)
            ax = fig.add_subplot(111)

        if data_labels is None:
            data_labels = np.ones(self.num_data)
        ulabels = []
        for lab in data_labels:
            if not lab in ulabels:
                ulabels.append(lab)
        marker = itertools.cycle(list(data_marker))
        from GPy.util import Tango
        for i, ul in enumerate(ulabels):
            if type(ul) is np.string_:
                this_label = ul
            elif type(ul) is np.int64:
                this_label = 'class %i' % ul
            else:
                this_label = 'class %i' % i
            m = marker.next()
            index = np.nonzero(data_labels == ul)[0]
            x = self.X[index, input_1]
            y = self.X[index, input_2]
            ax.scatter(x, y, marker=m, s=data_s, color=Tango.nextMedium(), label=this_label)

        ax.set_xlabel('latent dimension %i' % input_1)
        ax.set_ylabel('latent dimension %i' % input_2)

        from matplotlib.cm import get_cmap
        from GPy.util.latent_space_visualizations.controllers.imshow_controller import ImAnnotateController
        if not 'cmap' in kwargs.keys():
            kwargs.update(cmap=get_cmap('jet'))
        controller = ImAnnotateController(ax,
                                      plot_function,
                                      tuple(self.X.min(0)[:, significant_dims]) + tuple(self.X.max(0)[:, significant_dims]),
                                      resolution=resolution,
                                      aspect=aspect,
                                      **kwargs)
        ax.legend()
        ax.figure.tight_layout()
        if updates:
            pyplot.show()
            clear = raw_input('Enter to continue')
            if clear.lower() in 'yes' or clear == '':
                controller.deactivate()
        return controller.view

    def plot_X_1d(self, fignum=None, ax=None, colors=None):
        """
        Plot latent space X in 1D:

            - if fig is given, create input_dim subplots in fig and plot in these
            - if ax is given plot input_dim 1D latent space plots of X into each `axis`
            - if neither fig nor ax is given create a figure with fignum and plot in there

        colors:
            colors of different latent space dimensions input_dim

        """
        import pylab
        if ax is None:
            fig = pylab.figure(num=fignum, figsize=(8, min(12, (2 * self.X.shape[1]))))
        if colors is None:
            colors = pylab.gca()._get_lines.color_cycle
            pylab.clf()
        else:
            colors = iter(colors)
        plots = []
        x = np.arange(self.X.shape[0])
        for i in range(self.X.shape[1]):
            if ax is None:
                a = fig.add_subplot(self.X.shape[1], 1, i + 1)
            elif isinstance(ax, (tuple, list)):
                a = ax[i]
            else:
                raise ValueError("Need one ax per latent dimnesion input_dim")
            a.plot(self.X, c='k', alpha=.3)
            plots.extend(a.plot(x, self.X.T[i], c=colors.next(), label=r"$\mathbf{{X_{{{}}}}}$".format(i)))
            a.fill_between(x,
                            self.X.T[i] - 2 * np.sqrt(self.X_variance.T[i]),
                            self.X.T[i] + 2 * np.sqrt(self.X_variance.T[i]),
                            facecolor=plots[-1].get_color(),
                            alpha=.3)
            a.legend(borderaxespad=0.)
            a.set_xlim(x.min(), x.max())
            if i < self.X.shape[1] - 1:
                a.set_xticklabels('')
        pylab.draw()
        if ax is None:
            fig.tight_layout(h_pad=.01) # , rect=(0, 0, 1, .95))
        return fig

    def getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return SparseGP._getstate(self) + [self.init]

    def setstate(self, state):
        self._const_jitter = None
        self.init = state.pop()
        SparseGP._setstate(self, state)


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


