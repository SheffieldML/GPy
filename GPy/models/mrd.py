# ## Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
import pylab

from ..core import Model
from ..util.linalg import PCA
from ..kern import Kern
from ..core.parameterization.variational import NormalPosterior, NormalPrior
from ..core.parameterization import Param, Parameterized
from ..inference.latent_function_inference.var_dtc import VarDTCMissingData, VarDTC
from ..likelihoods import Gaussian

class MRD(Model):
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
    :type kernel: [GPy.kern.kern] | GPy.kern.kern | None (default)
    :param :class:`~GPy.inference.latent_function_inference inference_method: the inference method to use
    :param :class:`~GPy.likelihoods.likelihood.Likelihood` likelihood: the likelihood to use
    :param str name: the name of this model
    :param [str] Ynames: the names for the datasets given, must be of equal length as Ylist or None
    """
    
    def __init__(self, Ylist, input_dim, X=None, X_variance=None, 
                 initx = 'PCA', initz = 'permute',
                 num_inducing=10, Z=None, kernel=None, 
                 inference_method=None, likelihood=None, name='mrd', Ynames=None):
        super(MRD, self).__init__(name)
        
        # sort out the kernels
        if kernel is None:
            from ..kern import RBF
            self.kern = [RBF(input_dim, ARD=1, name='rbf'.format(i)) for i in range(len(Ylist))]
        elif isinstance(kernel, Kern):
            self.kern = [kernel.copy(name='{}'.format(kernel.name, i)) for i in range(len(Ylist))]
        else:
            assert len(kernel) == len(Ylist), "need one kernel per output"
            assert all([isinstance(k, Kern) for k in kernel]), "invalid kernel object detected!"
            self.kern = kernel
        self.input_dim = input_dim
        self.num_inducing = num_inducing
        
        self.Ylist = Ylist
        self._in_init_ = True
        X = self._init_X(initx, Ylist)
        self.Z = Param('inducing inputs', self._init_Z(initz, X))
        self.num_inducing = self.Z.shape[0] # ensure M==N if M>N
        
        if X_variance is None:
            X_variance = np.random.uniform(0, .2, X.shape)
        
        self.variational_prior = NormalPrior()
        self.X = NormalPosterior(X, X_variance)
        
        if likelihood is None:
            self.likelihood = [Gaussian(name='Gaussian_noise'.format(i)) for i in range(len(Ylist))]
        else: self.likelihood = likelihood
        
        if inference_method is None:
            self.inference_method= []
            for y in Ylist:
                if np.any(np.isnan(y)):
                    self.inference_method.append(VarDTCMissingData(limit=1))
                else:
                    self.inference_method.append(VarDTC(limit=1))
        else:
            self.inference_method = inference_method
            self.inference_method.set_limit(len(Ylist))
                
        self.add_parameters(self.X, self.Z)
        
        if Ynames is None:
            Ynames = ['Y{}'.format(i) for i in range(len(Ylist))]
        
        for i, n, k, l in itertools.izip(itertools.count(), Ynames, self.kern, self.likelihood):
            p = Parameterized(name=n)
            p.add_parameter(k)
            p.add_parameter(l)
            setattr(self, 'Y{}'.format(i), p)
            self.add_parameter(p)
        self._in_init_ = False

    def parameters_changed(self):
        self._log_marginal_likelihood = 0
        self.posteriors = []
        self.Z.gradient = 0.
        self.X.mean.gradient = 0.
        self.X.variance.gradient = 0.

        for y, k, l, i in itertools.izip(self.Ylist, self.kern, self.likelihood, self.inference_method):
            posterior, lml, grad_dict = i.inference(k, self.X, self.Z, l, y)

            self.posteriors.append(posterior)
            self._log_marginal_likelihood += lml

            # likelihood gradients
            l.update_gradients(grad_dict.pop('partial_for_likelihood'))

            #gradients wrt kernel
            dL_dKmm = grad_dict.pop('dL_dKmm')
            k.update_gradients_full(dL_dKmm, self.Z, None)
            target = k.gradient.copy()
            k.update_gradients_expectations(variational_posterior=self.X, Z=self.Z, **grad_dict)
            k.gradient += target

            #gradients wrt Z
            self.Z.gradient += k.gradients_X(dL_dKmm, self.Z)
            self.Z.gradient += k.gradients_Z_expectations(
                               grad_dict['dL_dpsi1'], grad_dict['dL_dpsi2'], Z=self.Z, variational_posterior=self.X)

            dL_dmean, dL_dS = k.gradients_qX_expectations(variational_posterior=self.X, Z=self.Z, **grad_dict)
            self.X.mean.gradient += dL_dmean
            self.X.variance.gradient += dL_dS

        # update for the KL divergence
        self.variational_prior.update_gradients_KL(self.X)
        self._log_marginal_likelihood -= self.variational_prior.KL_divergence(self.X)

    def log_likelihood(self):
        return self._log_marginal_likelihood

    def _init_X(self, init='PCA', Ylist=None):
        if Ylist is None:
            Ylist = self.Ylist
        if init in "PCA_concat":
            X = PCA(np.hstack(Ylist), self.input_dim)[0]
        elif init in "PCA_single":
            X = np.zeros((Ylist[0].shape[0], self.input_dim))
            for qs, Y in itertools.izip(np.array_split(np.arange(self.input_dim), len(Ylist)), Ylist):
                X[:, qs] = PCA(Y, len(qs))[0]
        else: # init == 'random':
            X = np.random.randn(Ylist[0].shape[0], self.input_dim)
        return X

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
            plotf(i, g, ax)
            if sharey_ax is not None:
                pylab.setp(ax.get_yticklabels(), visible=False)
        pylab.draw()
        if axes is None:
            fig.tight_layout()
            return fig
        else:
            return pylab.gcf()

    def plot_X(self, fignum=None, ax=None):
        fig = self._handle_plotting(fignum, ax, lambda i, g, ax: ax.imshow(g.X))
        return fig

    def plot_predict(self, fignum=None, ax=None, sharex=False, sharey=False, **kwargs):
        fig = self._handle_plotting(fignum,
                                    ax,
                                    lambda i, g, ax: ax.imshow(g. predict(g.X)[0], **kwargs),
                                    sharex=sharex, sharey=sharey)
        return fig

    def plot_scales(self, fignum=None, ax=None, titles=None, sharex=False, sharey=True, *args, **kwargs):
        """

        TODO: Explain other parameters

        :param titles: titles for axes of datasets

        """
        if titles is None:
            titles = [r'${}$'.format(name) for name in self.names]
        ymax = reduce(max, [np.ceil(max(g.input_sensitivity())) for g in self.bgplvms])
        def plotf(i, g, ax):
            ax.set_ylim([0,ymax])
            g.kern.plot_ARD(ax=ax, title=titles[i], *args, **kwargs)
        fig = self._handle_plotting(fignum, ax, plotf, sharex=sharex, sharey=sharey)
        return fig

    def plot_latent(self, fignum=None, ax=None, *args, **kwargs):
        fig = self.gref.plot_latent(fignum=fignum, ax=ax, *args, **kwargs) # self._handle_plotting(fignum, ax, lambda i, g, ax: g.plot_latent(ax=ax, *args, **kwargs))
        return fig

    def _debug_plot(self):
        self.plot_X_1d()
        fig = pylab.figure("MRD DEBUG PLOT", figsize=(4 * len(self.bgplvms), 9))
        fig.clf()
        axes = [fig.add_subplot(3, len(self.bgplvms), i + 1) for i in range(len(self.bgplvms))]
        self.plot_X(ax=axes)
        axes = [fig.add_subplot(3, len(self.bgplvms), i + len(self.bgplvms) + 1) for i in range(len(self.bgplvms))]
        self.plot_latent(ax=axes)
        axes = [fig.add_subplot(3, len(self.bgplvms), i + 2 * len(self.bgplvms) + 1) for i in range(len(self.bgplvms))]
        self.plot_scales(ax=axes)
        pylab.draw()
        fig.tight_layout()


