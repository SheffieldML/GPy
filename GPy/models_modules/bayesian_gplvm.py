# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from matplotlib import pyplot

from ..core.sparse_gp import SparseGP
from ..likelihoods import Gaussian
from .. import kern
from ..inference.optimization import SCG
from ..util import plot_latent, linalg
from .gplvm import GPLVM, initialise_latent
from ..util.plot_latent import most_significant_input_dimensions
from ..core.model import Model
from ..util.subarray_and_sorting import common_subarrays

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
    def __init__(self, likelihood_or_Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, **kwargs):
        if type(likelihood_or_Y) is np.ndarray:
            likelihood = Gaussian(likelihood_or_Y)
        else:
            likelihood = likelihood_or_Y

        if X == None:
            X = initialise_latent(init, input_dim, likelihood.Y)
        self.init = init

        if X_variance is None:
            X_variance = np.clip((np.ones_like(X) * 0.5) + .01 * np.random.randn(*X.shape), 0.001, 1)

        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.rbf(input_dim) # + kern.white(input_dim)

        SparseGP.__init__(self, X, likelihood, kernel, Z=Z, X_variance=X_variance, **kwargs)
        self.ensure_default_constraints()

    def _get_param_names(self):
        X_names = sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        return (X_names + S_names + SparseGP._get_param_names(self))

    #def _get_print_names(self):
    #    return SparseGP._get_print_names(self)

    def _get_params(self):
        """
        Horizontally stacks the parameters in order to present them to the optimizer.
        The resulting 1-input_dim array has this structure:

        ===============================================================
        |       mu       |        S        |    Z    | theta |  beta  |
        ===============================================================

        """
        x = np.hstack((self.X.flatten(), self.X_variance.flatten(), SparseGP._get_params(self)))
        return x

    def _set_params(self, x, save_old=True, save_count=0):
        N, input_dim = self.num_data, self.input_dim
        self.X = x[:self.X.size].reshape(N, input_dim).copy()
        self.X_variance = x[(N * input_dim):(2 * N * input_dim)].reshape(N, input_dim).copy()
        SparseGP._set_params(self, x[(2 * N * input_dim):])

    def dKL_dmuS(self):
        dKL_dS = (1. - (1. / (self.X_variance))) * 0.5
        dKL_dmu = self.X
        return dKL_dmu, dKL_dS

    def dL_dmuS(self):
        dL_dmu_psi0, dL_dS_psi0 = self.kern.dpsi0_dmuS(self.dL_dpsi0, self.Z, self.X, self.X_variance)
        dL_dmu_psi1, dL_dS_psi1 = self.kern.dpsi1_dmuS(self.dL_dpsi1, self.Z, self.X, self.X_variance)
        dL_dmu_psi2, dL_dS_psi2 = self.kern.dpsi2_dmuS(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        dL_dmu = dL_dmu_psi0 + dL_dmu_psi1 + dL_dmu_psi2
        dL_dS = dL_dS_psi0 + dL_dS_psi1 + dL_dS_psi2

        return dL_dmu, dL_dS

    def KL_divergence(self):
        var_mean = np.square(self.X).sum()
        var_S = np.sum(self.X_variance - np.log(self.X_variance))
        return 0.5 * (var_mean + var_S) - 0.5 * self.input_dim * self.num_data

    def log_likelihood(self):
        ll = SparseGP.log_likelihood(self)
        kl = self.KL_divergence()
        return ll - kl

    def _log_likelihood_gradients(self):
        dKL_dmu, dKL_dS = self.dKL_dmuS()
        dL_dmu, dL_dS = self.dL_dmuS()
        d_dmu = (dL_dmu - dKL_dmu).flatten()
        d_dS = (dL_dS - dKL_dS).flatten()
        self.dbound_dmuS = np.hstack((d_dmu, d_dS))
        self.dbound_dZtheta = SparseGP._log_likelihood_gradients(self)
        return np.hstack((self.dbound_dmuS.flatten(), self.dbound_dZtheta))

    def plot_latent(self, plot_inducing=True, *args, **kwargs):
        return plot_latent.plot_latent(self, plot_inducing=plot_inducing, *args, **kwargs)

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
        return SparseGP.getstate(self) + [self.init]

    def setstate(self, state):
        self._const_jitter = None
        self.init = state.pop()
        SparseGP.setstate(self, state)

class BayesianGPLVMWithMissingData(Model):
    """
    Bayesian Gaussian Process Latent Variable Model with missing data support.
    NOTE: Missing data is assumed to be missing at random!
    
    This extension comes with a large memory and computing time deficiency.
    Use only if fraction of missing data at random is higher than 60%.
    Otherwise, try filtering data before using this extension.
    
    Y can hold missing data as given by `missing`, standard is :class:`~numpy.nan`.
    
    If likelihood is given for Y, this likelihood will be discarded, but the parameters
    of the likelihood will be taken. Also every effort of creating the same likelihood
    will be done.
     
    :param likelihood_or_Y: observed data (np.ndarray) or GPy.likelihood
    :type likelihood_or_Y: :class:`~numpy.ndarray` | :class:`~GPy.likelihoods.likelihood.likelihood` instance
    :param int input_dim: latent dimensionality
    :param init: initialisation method for the latent space
    :type init: 'PCA' | 'random'
    """
    def __init__(self, likelihood_or_Y, input_dim, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, **kwargs):
        #=======================================================================
        # Filter Y, such that same missing data is at same positions. 
        # If full rows are missing, delete them entirely!
        if type(likelihood_or_Y) is np.ndarray:
            Y = likelihood_or_Y
            likelihood = Gaussian
            params = 1.
            normalize=None
        else:
            Y = likelihood_or_Y.Y
            likelihood = likelihood_or_Y.__class__
            params = likelihood_or_Y._get_params()
            if isinstance(likelihood_or_Y, Gaussian):
                normalize = True
                scale = likelihood_or_Y._scale
                offset = likelihood_or_Y._offset
        # Get common subrows
        filter_ = np.isnan(Y)
        self.subarray_indices = common_subarrays(filter_,axis=1)
        likelihoods = [likelihood(Y[~np.array(v,dtype=bool),:][:,ind]) for v,ind in self.subarray_indices.iteritems()]
        for l in likelihoods:
            l._set_params(params)
            if normalize: # get normalization in common
                l._scale = scale
                l._offset = offset
        #=======================================================================

        if X == None:
            X = initialise_latent(init, input_dim, Y[:,np.any(np.isnan(Y),1)])
        self.init = init

        if X_variance is None:
            X_variance = np.clip((np.ones_like(X) * 0.5) + .01 * np.random.randn(*X.shape), 0.001, 1)

        if Z is None:
            Z = np.random.permutation(X.copy())[:num_inducing]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.rbf(input_dim) # + kern.white(input_dim)

        self.submodels = [BayesianGPLVM(l, input_dim, X, X_variance, init, num_inducing, Z, kernel) for l in likelihoods]
        self.gref = self.submodels[0] 
        #:type self.gref: BayesianGPLVM 
        self.ensure_default_constraints()

    def log_likelihood(self):
        ll = -self.gref.KL_divergence()
        for g in self.submodels:
            ll += SparseGP.log_likelihood(g)
        return ll

    def _log_likelihood_gradients(self):
        dLdmu, dLdS = reduce(lambda a, b: [a[0] + b[0], a[1] + b[1]], (g.dL_dmuS() for g in self.bgplvms))
        dKLmu, dKLdS = self.gref.dKL_dmuS()
        dLdmu -= dKLmu
        dLdS -= dKLdS
        dLdmuS = np.hstack((dLdmu.flatten(), dLdS.flatten())).flatten()
        dldzt1 = reduce(lambda a, b: a + b, (SparseGP._log_likelihood_gradients(g)[:self.gref.num_inducing*self.gref.input_dim] for g in self.submodels))

        return np.hstack((dLdmuS,
                             dldzt1,
                np.hstack([np.hstack([g.dL_dtheta(),
                                            g.likelihood._gradients(\
                                                partial=g.partial_for_likelihood)]) \
                              for g in self.submodels])))

    def getstate(self):
        return Model.getstate(self)+[self.submodels,self.subarray_indices]

    def setstate(self, state):
        self.subarray_indices = state.pop()
        self.submodels = state.pop()
        self.gref = self.submodels[0]
        Model.setstate(self, state)
        self._set_params(self._get_params())

    def _get_param_names(self):
        X_names = sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        return (X_names + S_names + SparseGP._get_param_names(self.gref))

    def _get_params(self):
        return self.gref._get_params()
    def _set_params(self, x):
        [g._set_params(x) for g in self.submodels]
    

    pass

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


