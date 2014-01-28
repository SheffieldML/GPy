'''
Created on 10 Apr 2013

@author: Max Zwiessele
'''
from GPy.core import Model
from GPy.core import SparseGP
from GPy.util.linalg import pca
import numpy
import itertools
import pylab
from ..kern import kern
from bayesian_gplvm import BayesianGPLVM

class MRD(Model):
    """
    Do MRD on given Datasets in Ylist.
    All Ys in likelihood_list are in [N x Dn], where Dn can be different per Yn,
    N must be shared across datasets though.

    :param likelihood_list: list of observed datasets (:py:class:`~GPy.likelihoods.gaussian.Gaussian` if not supplied directly)
    :type likelihood_list: [:py:class:`~GPy.likelihoods.likelihood.likelihood` | :py:class:`ndarray`]
    :param names: names for different gplvm models
    :type names: [str]
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param initx: initialisation method for the latent space :

        * 'concat' - pca on concatenation of all datasets
        * 'single' - Concatenation of pca on datasets, respectively
        * 'random' - Random draw from a normal

    :type initx: ['concat'|'single'|'random']
    :param initz: initialisation method for inducing inputs
    :type initz: 'permute'|'random'
    :param X: Initial latent space
    :param X_variance: Initial latent space variance
    :param Z: initial inducing inputs
    :param num_inducing: number of inducing inputs to use
    :param kernels: list of kernels or kernel shared for all BGPLVMS
    :type kernels: [GPy.kern.kern] | GPy.kern.kern | None (default)

    """
    def __init__(self, likelihood_or_Y_list, input_dim, num_inducing=10, names=None,
                 kernels=None, initx='PCA',
                 initz='permute', _debug=False, **kw):
        if names is None:
            self.names = ["{}".format(i) for i in range(len(likelihood_or_Y_list))]
        else:
            self.names = names
            assert len(names) == len(likelihood_or_Y_list), "one name per data set required"
        # sort out the kernels
        if kernels is None:
            kernels = [None] * len(likelihood_or_Y_list)
        elif isinstance(kernels, kern):
            kernels = [kernels.copy() for i in range(len(likelihood_or_Y_list))]
        else:
            assert len(kernels) == len(likelihood_or_Y_list), "need one kernel per output"
            assert all([isinstance(k, kern) for k in kernels]), "invalid kernel object detected!"
        assert not ('kernel' in kw), "pass kernels through `kernels` argument"

        self.input_dim = input_dim
        self._debug = _debug
        self.num_inducing = num_inducing

        self._init = True
        X = self._init_X(initx, likelihood_or_Y_list)
        Z = self._init_Z(initz, X)
        self.num_inducing = Z.shape[0] # ensure M==N if M>N

        self.bgplvms = [BayesianGPLVM(l, input_dim=input_dim, kernel=k, X=X, Z=Z, num_inducing=self.num_inducing, **kw) for l, k in zip(likelihood_or_Y_list, kernels)]
        del self._init

        self.gref = self.bgplvms[0]
        nparams = numpy.array([0] + [SparseGP._get_params(g).size - g.Z.size for g in self.bgplvms])
        self.nparams = nparams.cumsum()

        self.num_data = self.gref.num_data

        self.NQ = self.num_data * self.input_dim
        self.MQ = self.num_inducing * self.input_dim

        Model.__init__(self)
        self.ensure_default_constraints()

    @property
    def X(self):
        return self.gref.X
    @X.setter
    def X(self, X):
        try:
            self.propagate_param(X=X)
        except AttributeError:
            if not self._init:
                raise AttributeError("bgplvm list not initialized")
    @property
    def Z(self):
        return self.gref.Z
    @Z.setter
    def Z(self, Z):
        try:
            self.propagate_param(Z=Z)
        except AttributeError:
            if not self._init:
                raise AttributeError("bgplvm list not initialized")
    @property
    def X_variance(self):
        return self.gref.X_variance
    @X_variance.setter
    def X_variance(self, X_var):
        try:
            self.propagate_param(X_variance=X_var)
        except AttributeError:
            if not self._init:
                raise AttributeError("bgplvm list not initialized")
    @property
    def likelihood_list(self):
        return [g.likelihood.Y for g in self.bgplvms]
    @likelihood_list.setter
    def likelihood_list(self, likelihood_list):
        for g, Y in itertools.izip(self.bgplvms, likelihood_list):
            g.likelihood.Y = Y

    @property
    def auto_scale_factor(self):
        """
        set auto_scale_factor for all gplvms
        :param b: auto_scale_factor
        :type b:
        """
        return self.gref.auto_scale_factor
    @auto_scale_factor.setter
    def auto_scale_factor(self, b):
        self.propagate_param(auto_scale_factor=b)

    def propagate_param(self, **kwargs):
        for key, val in kwargs.iteritems():
            for g in self.bgplvms:
                g.__setattr__(key, val)

    def randomize(self, initx='concat', initz='permute', *args, **kw):
        super(MRD, self).randomize(*args, **kw)
        self._init_X(initx, self.likelihood_list)
        self._init_Z(initz, self.X)

    #def _get_latent_param_names(self):
    def _get_param_names(self):
        n1 = self.gref._get_param_names()
        n1var = n1[:self.NQ * 2 + self.MQ]
    #    return n1var
    #
    #def _get_kernel_names(self):
        map_names = lambda ns, name: map(lambda x: "{1}_{0}".format(*x),
                                         itertools.izip(ns,
                                                        itertools.repeat(name)))
        return list(itertools.chain(n1var, *(map_names(\
                SparseGP._get_param_names(g)[self.MQ:], n) \
                for g, n in zip(self.bgplvms, self.names))))
    #    kernel_names = (map_names(SparseGP._get_param_names(g)[self.MQ:], n) for g, n in zip(self.bgplvms, self.names))
    #    return kernel_names

    #def _get_param_names(self):
        # X_names = sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        # S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
    #    n1var = self._get_latent_param_names()
    #    kernel_names = self._get_kernel_names()
    #    return list(itertools.chain(n1var, *kernel_names))

    #def _get_print_names(self):
    #    return list(itertools.chain(*self._get_kernel_names()))

    def _get_params(self):
        """
        return parameter list containing private and shared parameters as follows:

        =================================================================
        | mu | S | Z || theta1 | theta2 | .. | thetaN |
        =================================================================
        """
        X = self.gref.X.ravel()
        X_var = self.gref.X_variance.ravel()
        Z = self.gref.Z.ravel()
        thetas = [SparseGP._get_params(g)[g.Z.size:] for g in self.bgplvms]
        params = numpy.hstack([X, X_var, Z, numpy.hstack(thetas)])
        return params

#     def _set_var_params(self, g, X, X_var, Z):
#         g.X = X.reshape(self.num_data, self.input_dim)
#         g.X_variance = X_var.reshape(self.num_data, self.input_dim)
#         g.Z = Z.reshape(self.num_inducing, self.input_dim)
#
#     def _set_kern_params(self, g, p):
#         g.kern._set_params(p[:g.kern.num_params])
#         g.likelihood._set_params(p[g.kern.num_params:])

    def _set_params(self, x):
        start = 0; end = self.NQ
        X = x[start:end]
        start = end; end += start
        X_var = x[start:end]
        start = end; end += self.MQ
        Z = x[start:end]
        thetas = x[end:]

        # set params for all:
        for g, s, e in itertools.izip(self.bgplvms, self.nparams, self.nparams[1:]):
            g._set_params(numpy.hstack([X, X_var, Z, thetas[s:e]]))
#             self._set_var_params(g, X, X_var, Z)
#             self._set_kern_params(g, thetas[s:e].copy())
#             g._compute_kernel_matrices()
#             if self.auto_scale_factor:
#                 g.scale_factor = numpy.sqrt(g.psi2.sum(0).mean() * g.likelihood.precision)
# #                 self.scale_factor = numpy.sqrt(self.psi2.sum(0).mean() * self.likelihood.precision)
#             g._computations()


    def update_likelihood_approximation(self): # TODO: object oriented vs script base
        for bgplvm in self.bgplvms:
            bgplvm.update_likelihood_approximation()

    def log_likelihood(self):
        ll = -self.gref.KL_divergence()
        for g in self.bgplvms:
            ll += SparseGP.log_likelihood(g)
        return ll

    def _log_likelihood_gradients(self):
        dLdmu, dLdS = reduce(lambda a, b: [a[0] + b[0], a[1] + b[1]], (g.dL_dmuS() for g in self.bgplvms))
        dKLmu, dKLdS = self.gref.dKL_dmuS()
        dLdmu -= dKLmu
        dLdS -= dKLdS
        dLdmuS = numpy.hstack((dLdmu.flatten(), dLdS.flatten())).flatten()
        dldzt1 = reduce(lambda a, b: a + b, (SparseGP._log_likelihood_gradients(g)[:self.MQ] for g in self.bgplvms))

        return numpy.hstack((dLdmuS,
                             dldzt1,
                numpy.hstack([numpy.hstack([g.dL_dtheta(),
                                            g.likelihood._gradients(\
                                                partial=g.partial_for_likelihood)]) \
                              for g in self.bgplvms])))

    def _init_X(self, init='PCA', likelihood_list=None):
        if likelihood_list is None:
            likelihood_list = self.likelihood_list
        Ylist = []
        for likelihood_or_Y in likelihood_list:
            if type(likelihood_or_Y) is numpy.ndarray:
                Ylist.append(likelihood_or_Y)
            else:
                Ylist.append(likelihood_or_Y.Y)
        del likelihood_list
        if init in "PCA_concat":
            X = pca(numpy.hstack(Ylist), self.input_dim)[0]
        elif init in "PCA_single":
            X = numpy.zeros((Ylist[0].shape[0], self.input_dim))
            for qs, Y in itertools.izip(numpy.array_split(numpy.arange(self.input_dim), len(Ylist)), Ylist):
                X[:, qs] = pca(Y, len(qs))[0]
        else: # init == 'random':
            X = numpy.random.randn(Ylist[0].shape[0], self.input_dim)
        self.X = X
        return X


    def _init_Z(self, init="permute", X=None):
        if X is None:
            X = self.X
        if init in "permute":
            Z = numpy.random.permutation(X.copy())[:self.num_inducing]
        elif init in "random":
            Z = numpy.random.randn(self.num_inducing, self.input_dim) * X.var()
        self.Z = Z
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

    def plot_X_1d(self, *a, **kw):
        return self.gref.plot_X_1d(*a, **kw)

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
        ymax = reduce(max, [numpy.ceil(max(g.input_sensitivity())) for g in self.bgplvms])
        def plotf(i, g, axis):
            axis.set_ylim([0,ymax])
            g.kern.plot_ARD(ax=axis, title=titles[i], *args, **kwargs)
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

    def getstate(self):
        return Model.getstate(self) + [self.names,
                self.bgplvms,
                self.gref,
                self.nparams,
                self.input_dim,
                self.num_inducing,
                self.num_data,
                self.NQ,
                self.MQ]

    def setstate(self, state):
        self.MQ = state.pop()
        self.NQ = state.pop()
        self.num_data = state.pop()
        self.num_inducing = state.pop()
        self.input_dim = state.pop()
        self.nparams = state.pop()
        self.gref = state.pop()
        self.bgplvms = state.pop()
        self.names = state.pop()
        Model.setstate(self, state)



