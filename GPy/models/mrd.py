# ## Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools, logging

from ..kern import Kern
from ..core.parameterization.variational import NormalPosterior, NormalPrior
from ..core.parameterization import Param, Parameterized
from ..core.parameterization.observable_array import ObsAr
from ..inference.latent_function_inference.var_dtc import VarDTC
from ..inference.latent_function_inference import InferenceMethodList
from ..likelihoods import Gaussian
from ..util.initialization import initialize_latent
from ..core.sparse_gp import SparseGP, GP
from GPy.core.parameterization.variational import VariationalPosterior
from GPy.models.bayesian_gplvm_minibatch import BayesianGPLVMMiniBatch
from GPy.models.sparse_gp_minibatch import SparseGPMiniBatch

class MRD(BayesianGPLVMMiniBatch):
    """
    !WARNING: This is bleeding edge code and still in development.
    Functionality may change fundamentally during development!

    Apply MRD to all given datasets Y in Ylist.

    Y_i in [n x p_i]

    If Ylist is a dictionary, the keys of the dictionary are the names, and the
    values are the different datasets to compare.

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
    :param bool|Norm normalizer: How to normalize the data?
    :param bool stochastic: Should this model be using stochastic gradient descent over the dimensions?
    :param bool|[bool] batchsize: either one batchsize for all, or one batchsize per dataset.
    """
    def __init__(self, Ylist, input_dim, X=None, X_variance=None,
                 initx = 'PCA', initz = 'permute',
                 num_inducing=10, Z=None, kernel=None,
                 inference_method=None, likelihoods=None, name='mrd',
                 Ynames=None, normalizer=False, stochastic=False, batchsize=10):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_dim = input_dim
        self.num_inducing = num_inducing

        if isinstance(Ylist, dict):
            Ynames, Ylist = zip(*Ylist.items())

        self.logger.debug("creating observable arrays")
        self.Ylist = [ObsAr(Y) for Y in Ylist]
        #The next line is a fix for Python 3. It replicates the python 2 behaviour from the above comprehension
        Y = Ylist[-1]

        if Ynames is None:
            self.logger.debug("creating Ynames")
            Ynames = ['Y{}'.format(i) for i in range(len(Ylist))]
        self.names = Ynames
        assert len(self.names) == len(self.Ylist), "one name per dataset, or None if Ylist is a dict"

        if inference_method is None:
            self.inference_method = InferenceMethodList([VarDTC() for _ in range(len(self.Ylist))])
        else:
            assert isinstance(inference_method, InferenceMethodList), "please provide one inference method per Y in the list and provide it as InferenceMethodList, inference_method given: {}".format(inference_method)
            self.inference_method = inference_method

        if X is None:
            X, fracs = self._init_X(initx, Ylist)
        else:
            fracs = [X.var(0)]*len(Ylist)

        Z = self._init_Z(initz, X)
        self.Z = Param('inducing inputs', Z)
        self.num_inducing = self.Z.shape[0] # ensure M==N if M>N

        # sort out the kernels
        self.logger.info("building kernels")
        if kernel is None:
            from ..kern import RBF
            kernels = [RBF(input_dim, ARD=1, lengthscale=1./fracs[i]) for i in range(len(Ylist))]
        elif isinstance(kernel, Kern):
            kernels = []
            for i in range(len(Ylist)):
                k = kernel.copy()
                kernels.append(k)
        else:
            assert len(kernel) == len(Ylist), "need one kernel per output"
            assert all([isinstance(k, Kern) for k in kernel]), "invalid kernel object detected!"
            kernels = kernel

        self.variational_prior = NormalPrior()
        #self.X = NormalPosterior(X, X_variance)

        if likelihoods is None:
            likelihoods = [Gaussian(name='Gaussian_noise'.format(i)) for i in range(len(Ylist))]
        else: likelihoods = likelihoods

        self.logger.info("adding X and Z")
        super(MRD, self).__init__(Y, input_dim, X=X, X_variance=X_variance, num_inducing=num_inducing,
                 Z=self.Z, kernel=None, inference_method=self.inference_method, likelihood=Gaussian(),
                 name='manifold relevance determination', normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1)

        self._log_marginal_likelihood = 0

        self.unlink_parameter(self.likelihood)
        self.unlink_parameter(self.kern)
        del self.kern
        del self.likelihood

        self.num_data = Ylist[0].shape[0]
        if isinstance(batchsize, int):
            batchsize = itertools.repeat(batchsize)

        self.bgplvms = []

        for i, n, k, l, Y, im, bs in zip(itertools.count(), Ynames, kernels, likelihoods, Ylist, self.inference_method, batchsize):
            assert Y.shape[0] == self.num_data, "All datasets need to share the number of datapoints, and those have to correspond to one another"
            md = np.isnan(Y).any()
            spgp = BayesianGPLVMMiniBatch(Y, input_dim, X, X_variance,
                                          Z=Z, kernel=k, likelihood=l,
                                          inference_method=im, name=n,
                                          normalizer=normalizer,
                                          missing_data=md,
                                          stochastic=stochastic,
                                          batchsize=bs)
            spgp.kl_factr = 1./len(Ynames)
            spgp.unlink_parameter(spgp.Z)
            spgp.unlink_parameter(spgp.X)
            del spgp.Z
            del spgp.X
            spgp.Z = self.Z
            spgp.X = self.X
            self.link_parameter(spgp, i+2)
            self.bgplvms.append(spgp)

        self.posterior = None
        self.logger.info("init done")

    def parameters_changed(self):
        self._log_marginal_likelihood = 0
        self.Z.gradient[:] = 0.
        self.X.gradient[:] = 0.
        for b, i in zip(self.bgplvms, self.inference_method):
            self._log_marginal_likelihood += b._log_marginal_likelihood

            self.logger.info('working on im <{}>'.format(hex(id(i))))
            self.Z.gradient[:] += b.Z.gradient#full_values['Zgrad']
            #grad_dict = b.full_values

            if self.has_uncertain_inputs():
                self.X.gradient += b._Xgrad
            else:
                self.X.gradient += b._Xgrad

        #if self.has_uncertain_inputs():
        #    # update for the KL divergence
        #    self.variational_prior.update_gradients_KL(self.X)
        #    self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)
        #    pass

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _init_X(self, init='PCA', Ylist=None):
        if Ylist is None:
            Ylist = self.Ylist
        if init in "PCA_concat":
            X, fracs = initialize_latent('PCA', self.input_dim, np.hstack(Ylist))
            fracs = [fracs]*len(Ylist)
        elif init in "PCA_single":
            X = np.zeros((Ylist[0].shape[0], self.input_dim))
            fracs = []
            for qs, Y in zip(np.array_split(np.arange(self.input_dim), len(Ylist)), Ylist):
                x,frcs = initialize_latent('PCA', len(qs), Y)
                X[:, qs] = x
                fracs.append(frcs)
        else: # init == 'random':
            X = np.random.randn(Ylist[0].shape[0], self.input_dim)
            fracs = X.var(0)
            fracs = [fracs]*len(Ylist)
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
        import matplotlib.pyplot as plt
        if axes is None:
            fig = plt.figure(num=fignum)
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
            elif isinstance(axes, (tuple, list, np.ndarray)):
                ax = axes[i]
            else:
                raise ValueError("Need one axes per latent dimension input_dim")
            plots.append(plotf(i, g, ax))
            if sharey_ax is not None:
                plt.setp(ax.get_yticklabels(), visible=False)
        plt.draw()
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
        b = self.bgplvms[Yindex]
        self.posterior = b.posterior
        self.kern = b.kern
        self.likelihood = b.likelihood
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
            #ax.set_ylim([0,ymax])
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
        from matplotlib import pyplot as plt
        from ..plotting.matplot_dep import dim_reduction_plots
        if "Yindex" not in predict_kwargs:
            predict_kwargs['Yindex'] = 0

        Yindex = predict_kwargs['Yindex']
        if ax is None:
            fig = plt.figure(num=fignum)
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        self.kern = self.bgplvms[Yindex].kern
        self.likelihood = self.bgplvms[Yindex].likelihood
        plot = dim_reduction_plots.plot_latent(self, labels, which_indices,
                                        resolution, ax, marker, s,
                                        fignum, plot_inducing, legend,
                                        plot_limits, aspect, updates, predict_kwargs, imshow_kwargs)
        ax.set_title(self.bgplvms[Yindex].name)
        try:
            fig.tight_layout()
        except:
            pass

        return plot

    def __getstate__(self):
        state = super(MRD, self).__getstate__()
        if 'kern' in state:
            del state['kern']
        if 'likelihood' in state:
            del state['likelihood']
        return state

    def __setstate__(self, state):
        # TODO:
        super(MRD, self).__setstate__(state)
        self.kern = self.bgplvms[0].kern
        self.likelihood = self.bgplvms[0].likelihood
        self.parameters_changed()
