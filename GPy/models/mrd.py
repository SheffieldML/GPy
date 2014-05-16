# ## Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
import pylab

from ..core import Model
from ..kern import Kern
from ..core.parameterization.variational import NormalPosterior, NormalPrior
from ..core.parameterization import Param, Parameterized
from ..inference.latent_function_inference.var_dtc import VarDTCMissingData, VarDTC
from ..likelihoods import Gaussian
from ..util.initialization import initialize_latent
from ..core.sparse_gp import SparseGP, GP
from ..inference.latent_function_inference import InferenceMethodList

class MRD(SparseGP):
    """
    Apply MRD to all given datasets Y in Ylist.

    Y_i in [n x p_i]

    The samples n in the datasets need
    to match up, whereas the dimensionality p_d can differ.

    :param [array-like] Ylist: List of datasets to apply MRD on
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param array-like X: mean of starting latent space q in [n x q]
    :param array-like X_variance: variance of starting latent space q in [n x q]
    :param initx: initialisation method for the latent space :

        * 'concat' - PCA on concatenation of all datasets
        * 'single' - Concatenation of PCA on datasets, respectively
        * 'random' - Random draw from a Normal(0,1)

    :type initx: ['concat'|'single'|'random']
    :param initz: initialisation method for inducing inputs
    :type initz: 'permute'|'random'
    :param num_inducing: number of inducing inputs to use
    :param Z: initial inducing inputs
    :param kernel: list of kernels or kernel to copy for each output
    :type kernel: [GPy.kernels.kernels] | GPy.kernels.kernels | None (default)
    :param :class:`~GPy.inference.latent_function_inference inference_method: 
        InferenceMethodList of inferences, or one inference method for all
    :param :class:`~GPy.likelihoodss.likelihoods.likelihoods` likelihoods: the likelihoods to use
    :param str name: the name of this model
    :param [str] Ynames: the names for the datasets given, must be of equal length as Ylist or None
    """
    def __init__(self, Ylist, input_dim, X=None, X_variance=None,
                 initx = 'PCA', initz = 'permute',
                 num_inducing=10, Z=None, kernel=None,
                 inference_method=None, likelihoods=None, name='mrd', Ynames=None):
        super(GP, self).__init__(name)

        self.input_dim = input_dim
        self.num_inducing = num_inducing

        self.Ylist = Ylist
        self._in_init_ = True
        X, fracs = self._init_X(initx, Ylist)
        self.Z = Param('inducing inputs', self._init_Z(initz, X))
        self.num_inducing = self.Z.shape[0] # ensure M==N if M>N

        # sort out the kernels
        if kernel is None:
            from ..kern import RBF
            self.kernels = [RBF(input_dim, ARD=1, lengthscale=fracs[i], name='rbf'.format(i)) for i in range(len(Ylist))]
        elif isinstance(kernel, Kern):
            self.kernels = []
            for i in range(len(Ylist)):
                k = kernel.copy()
                self.kernels.append(k)
        else:
            assert len(kernel) == len(Ylist), "need one kernel per output"
            assert all([isinstance(k, Kern) for k in kernel]), "invalid kernel object detected!"
            self.kernels = kernel

        if X_variance is None:
            X_variance = np.random.uniform(0.1, 0.2, X.shape)

        self.variational_prior = NormalPrior()
        self.X = NormalPosterior(X, X_variance)

        if likelihoods is None:
            self.likelihoods = [Gaussian(name='Gaussian_noise'.format(i)) for i in range(len(Ylist))]
        else: self.likelihoods = likelihoods

        if inference_method is None:
            self.inference_method= InferenceMethodList()
            for y in Ylist:
                inan = np.isnan(y)
                if np.any(inan):
                    self.inference_method.append(VarDTCMissingData(limit=1, inan=inan))
                else:
                    self.inference_method.append(VarDTC(limit=1))
        else:
            if not isinstance(inference_method, InferenceMethodList):
                inference_method = InferenceMethodList(inference_method)
            self.inference_method = inference_method

        self.add_parameters(self.X, self.Z)

        if Ynames is None:
            Ynames = ['Y{}'.format(i) for i in range(len(Ylist))]
        self.names = Ynames

        self.bgplvms = []
        self.num_data = Ylist[0].shape[0]

        for i, n, k, l, Y in itertools.izip(itertools.count(), Ynames, self.kernels, self.likelihoods, self.Ylist):
            assert Y.shape[0] == self.num_data, "All datasets need to share the number of datapoints, and those have to correspond to one another"

            p = Parameterized(name=n)
            p.add_parameter(k)
            p.kern = k
            p.add_parameter(l)
            p.likelihood = l
            self.add_parameter(p)
            self.bgplvms.append(p)

        self.posterior = None
        self._in_init_ = False

    def parameters_changed(self):
        self._log_marginal_likelihood = 0
        self.posteriors = []
        self.Z.gradient[:] = 0.
        self.X.gradient[:] = 0.

        for y, k, l, i in itertools.izip(self.Ylist, self.kernels, self.likelihoods, self.inference_method):
            posterior, lml, grad_dict = i.inference(k, self.X, self.Z, l, y)

            self.posteriors.append(posterior)
            self._log_marginal_likelihood += lml

            # likelihoods gradients
            l.update_gradients(grad_dict.pop('dL_dthetaL'))

            #gradients wrt kernel
            dL_dKmm = grad_dict.pop('dL_dKmm')
            k.update_gradients_full(dL_dKmm, self.Z, None)
            target = k.gradient.copy()
            k.update_gradients_expectations(variational_posterior=self.X, Z=self.Z, **grad_dict)
            k.gradient += target

            #gradients wrt Z
            self.Z.gradient += k.gradients_X(dL_dKmm, self.Z)
            self.Z.gradient += k.gradients_Z_expectations(
                               grad_dict['dL_dpsi0'], 
                               grad_dict['dL_dpsi1'], 
                               grad_dict['dL_dpsi2'], 
                               Z=self.Z, variational_posterior=self.X)

            dL_dmean, dL_dS = k.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, **grad_dict)
            self.X.mean.gradient += dL_dmean
            self.X.variance.gradient += dL_dS

        # update for the KL divergence
        self.posterior = self.posteriors[0]
        self.kern = self.kernels[0]
        self.likelihood = self.likelihoods[0]

        self.variational_prior.update_gradients_KL(self.X)
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _init_X(self, init='PCA', Ylist=None):
        if Ylist is None:
            Ylist = self.Ylist
        if init in "PCA_concat":
            X, fracs = initialize_latent('PCA', self.input_dim, np.hstack(Ylist))
            fracs = [fracs]*self.input_dim
        elif init in "PCA_single":
            X = np.zeros((Ylist[0].shape[0], self.input_dim))
            fracs = []
            for qs, Y in itertools.izip(np.array_split(np.arange(self.input_dim), len(Ylist)), Ylist):
                x,frcs = initialize_latent('PCA', len(qs), Y)
                X[:, qs] = x
                fracs.append(frcs)
        else: # init == 'random':
            X = np.random.randn(Ylist[0].shape[0], self.input_dim)
            fracs = X.var(0)
            fracs = [fracs]*self.input_dim
        X -= X.mean()
        X /= X.std()
        return X, fracs

    def _init_Z(self, init="permute", X=None):
        if X is None:
            X = self.X
        if init in "permute":
            Z = np.random.permutation(X.copy())[:self.num_inducing]
        elif init in "random":
            Z = np.random.randn(self.num_inducing, self.input_dim) * X.var()
        return Z

    def _handle_plotting(self, fignum, axes, plotf, sharex=False, sharey=False):
        if axes is None:
            fig = pylab.figure(num=fignum)
        sharex_ax = None
        sharey_ax = None
        plots = []
        for i, g in enumerate(self.bgplvms):
            try:
                if sharex:
                    sharex_ax = ax # @UndefinedVariable
                    sharex = False # dont set twice
                if sharey:
                    sharey_ax = ax # @UndefinedVariable
                    sharey = False # dont set twice
            except:
                pass
            if axes is None:
                ax = fig.add_subplot(1, len(self.bgplvms), i + 1, sharex=sharex_ax, sharey=sharey_ax)
            elif isinstance(axes, (tuple, list)):
                ax = axes[i]
            else:
                raise ValueError("Need one axes per latent dimension input_dim")
            plots.append(plotf(i, g, ax))
            if sharey_ax is not None:
                pylab.setp(ax.get_yticklabels(), visible=False)
        pylab.draw()
        if axes is None:
            try:
                fig.tight_layout()
            except:
                pass
        return plots

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, Yindex=0):
        """
        Prediction for data set Yindex[default=0].
        This predicts the output mean and variance for the dataset given in Ylist[Yindex]
        """
        self.posterior = self.posteriors[Yindex]
        self.kern = self.kernels[Yindex]
        self.likelihood = self.likelihoods[Yindex]
        return super(MRD, self).predict(Xnew, full_cov, Y_metadata, kern)

    #===============================================================================
    # TODO: Predict! Maybe even change to several bgplvms, which share an X?
    #===============================================================================
    #     def plot_predict(self, fignum=None, ax=None, sharex=False, sharey=False, **kwargs):
    #         fig = self._handle_plotting(fignum,
    #                                     ax,
    #                                     lambda i, g, ax: ax.imshow(g.predict(g.X)[0], **kwargs),
    #                                     sharex=sharex, sharey=sharey)
    #         return fig

    def plot_scales(self, fignum=None, ax=None, titles=None, sharex=False, sharey=True, *args, **kwargs):
        """

        TODO: Explain other parameters

        :param titles: titles for axes of datasets

        """
        if titles is None:
            titles = [r'${}$'.format(name) for name in self.names]
        ymax = reduce(max, [np.ceil(max(g.kern.input_sensitivity())) for g in self.bgplvms])
        def plotf(i, g, ax):
            ax.set_ylim([0,ymax])
            return g.kern.plot_ARD(ax=ax, title=titles[i], *args, **kwargs)
        fig = self._handle_plotting(fignum, ax, plotf, sharex=sharex, sharey=sharey)
        return fig

    def plot_latent(self, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=True, legend=True,
                plot_limits=None, 
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
        """
        see plotting.matplot_dep.dim_reduction_plots.plot_latent
        if predict_kwargs is None, will plot latent spaces for 0th dataset (and kernel), otherwise give
        predict_kwargs=dict(Yindex='index') for plotting only the latent space of dataset with 'index'.
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import dim_reduction_plots
        if "Yindex" not in predict_kwargs:
            predict_kwargs['Yindex'] = 0
        if ax is None:
            fig = pylab.figure(num=fignum)
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        plot = dim_reduction_plots.plot_latent(self, labels, which_indices,
                                        resolution, ax, marker, s,
                                        fignum, plot_inducing, legend,
                                        plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)
        ax.set_title(self.bgplvms[predict_kwargs['Yindex']].name)
        try:
            fig.tight_layout()
        except:
            pass

        return plot

    def __getstate__(self):
        # TODO:
        import copy
        state = copy.copy(self.__dict__)
        del state['kernels']
        del state['kern']
        del state['likelihood']
        return state

    def __setstate__(self, state):
        # TODO:
        super(MRD, self).__setstate__(state)
        self.kernels = [p.kern for p in self.bgplvms]
        self.kern = self.kernels[0]
        self.likelihood = self.likelihoods[0]
        self.parameters_changed()