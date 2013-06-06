'''
Created on 10 Apr 2013

@author: Max Zwiessele
'''
from GPy.core import Model
from GPy.core import SparseGP
from GPy.util.linalg import PCA
import numpy
import itertools
import pylab
from GPy.kern.kern import kern
from GPy.models.bayesian_gplvm import BayesianGPLVM

class MRD(Model):
    """
    Do MRD on given Datasets in Ylist.
    All Ys in likelihood_list are in [N x Dn], where Dn can be different per Yn,
    N must be shared across datasets though.

    :param likelihood_list...: likelihoods of observed datasets
    :type likelihood_list: [GPy.likelihood] | [Y1..Yy]
    :param names: names for different gplvm models
    :type names: [str]
    :param input_dim: latent dimensionality (will raise
    :type input_dim: int
    :param initx: initialisation method for the latent space
    :type initx: 'PCA'|'random'
    :param initz: initialisation method for inducing inputs
    :type initz: 'permute'|'random'
    :param X:
        Initial latent space
    :param X_variance:
        Initial latent space variance
    :param init: [cooncat|single|random]
        initialization method to use:
            *concat: PCA on concatenated outputs
            *single: PCA on each output
            *random: random
    :param num_inducing:
        number of inducing inputs to use
    :param Z:
        initial inducing inputs
    :param kernels: list of kernels or kernel shared for all BGPLVMS
    :type kernels: [GPy.kern.kern] | GPy.kern.kern | None (default)
    """
    def __init__(self, likelihood_or_Y_list, input_dim, num_inducing=10, names=None,
                 kernels=None, initx='PCA',
                 initz='permute', _debug=False, **kw):
        if names is None:
            self.names = ["{}".format(i + 1) for i in range(len(likelihood_or_Y_list))]

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
        self.num_inducing = num_inducing
        self._debug = _debug

        self._init = True
        X = self._init_X(initx, likelihood_or_Y_list)
        Z = self._init_Z(initz, X)
        self.bgplvms = [BayesianGPLVM(l, input_dim=input_dim, kernel=k, X=X, Z=Z, num_inducing=self.num_inducing, **kw) for l, k in zip(likelihood_or_Y_list, kernels)]
        del self._init

        self.gref = self.bgplvms[0]
        nparams = numpy.array([0] + [SparseGP._get_params(g).size - g.Z.size for g in self.bgplvms])
        self.nparams = nparams.cumsum()

        self.num_data = self.gref.num_data
        self.NQ = self.num_data * self.input_dim
        self.MQ = self.num_inducing * self.input_dim

        Model.__init__(self)
        self._set_params(self._get_params())

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

    def _get_param_names(self):
        # X_names = sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        # S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], [])
        n1 = self.gref._get_param_names()
        n1var = n1[:self.NQ * 2 + self.MQ]
        map_names = lambda ns, name: map(lambda x: "{1}_{0}".format(*x),
                                         itertools.izip(ns,
                                                        itertools.repeat(name)))
        return list(itertools.chain(n1var, *(map_names(\
                SparseGP._get_param_names(g)[self.MQ:], n) \
                for g, n in zip(self.bgplvms, self.names))))

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
#         g.kern._set_params(p[:g.kern.Nparam])
#         g.likelihood._set_params(p[g.kern.Nparam:])

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
            X = PCA(numpy.hstack(Ylist), self.input_dim)[0]
        elif init in "PCA_single":
            X = numpy.zeros((Ylist[0].shape[0], self.input_dim))
            for qs, Y in itertools.izip(numpy.array_split(numpy.arange(self.input_dim), len(Ylist)), Ylist):
                X[:, qs] = PCA(Y, len(qs))[0]
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

    def _handle_plotting(self, fignum, axes, plotf):
        if axes is None:
            fig = pylab.figure(num=fignum, figsize=(4 * len(self.bgplvms), 3))
        for i, g in enumerate(self.bgplvms):
            if axes is None:
                ax = fig.add_subplot(1, len(self.bgplvms), i + 1)
            elif isinstance(axes, (tuple, list)):
                ax = axes[i]
            else:
                raise ValueError("Need one axes per latent dimension input_dim")
            plotf(i, g, ax)
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

    def plot_predict(self, fignum=None, ax=None, **kwargs):
        fig = self._handle_plotting(fignum, ax, lambda i, g, ax: ax.imshow(g. predict(g.X)[0], **kwargs))
        return fig

    def plot_scales(self, fignum=None, ax=None, *args, **kwargs):
        fig = self._handle_plotting(fignum, ax, lambda i, g, ax: g.kern.plot_ARD(ax=ax, *args, **kwargs))
        return fig

    def plot_latent(self, fignum=None, ax=None, *args, **kwargs):
        fig = self._handle_plotting(fignum, ax, lambda i, g, ax: g.plot_latent(ax=ax, *args, **kwargs))
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

    def _debug_optimize(self, opt='scg', maxiters=5000, itersteps=10):
        iters = 0
        optstep = lambda: self.optimize(opt, messages=1, max_f_eval=itersteps)
        self._debug_plot()
        raw_input("enter to start debug")
        while iters < maxiters:
            optstep()
            self._debug_plot()
            iters += itersteps

