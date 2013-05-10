'''
Created on 10 Apr 2013

@author: Max Zwiessele
'''
from GPy.core import model
from GPy.models.Bayesian_GPLVM import Bayesian_GPLVM
from GPy.models.sparse_GP import sparse_GP
from GPy.util.linalg import PCA
from scipy import linalg
import numpy
import itertools
import pylab

class MRD(model):
    """
    Do MRD on given Datasets in Ylist.
    All Ys in Ylist are in [N x Dn], where Dn can be different per Yn,
    N must be shared across datasets though.

    :param Ylist...: observed datasets
    :type Ylist: [np.ndarray]
    :param names: names for different gplvm models
    :type names: [str]
    :param Q: latent dimensionality (will raise 
    :type Q: int
    :param initx: initialisation method for the latent space
    :type initx: 'PCA'|'random'
    :param initz: initialisation method for inducing inputs
    :type initz: 'permute'|'random'
    :param X:
        Initial latent space
    :param X_variance:
        Initial latent space variance
    :param init: [PCA|random]
        initialization method to use
    :param M:
        number of inducing inputs to use
    :param Z:
        initial inducing inputs
    :param kernel:
        kernel to use
    """

    def __init__(self, *Ylist, **kwargs):
        if kwargs.has_key("_debug"):
            self._debug = kwargs['_debug']
            del kwargs['_debug']
        else:
            self._debug = False
        if kwargs.has_key("names"):
            self.names = kwargs['names']
            del kwargs['names']
        else:
            self.names = ["{}".format(i + 1) for i in range(len(Ylist))]
        if kwargs.has_key('kernel'):
            kernel = kwargs['kernel']
            k = lambda: kernel.copy()
            del kwargs['kernel']
        else:
            k = lambda: None
        if kwargs.has_key('initx'):
            initx = kwargs['initx']
            del kwargs['initx']
        else:
            initx = "PCA"
        if kwargs.has_key('initz'):
            initz = kwargs['initz']
            del kwargs['initz']
        else:
            initz = "permute"
        try:
            self.Q = kwargs["Q"]
        except KeyError:
            raise ValueError("Need Q for MRD")
        try:
            self.M = kwargs["M"]
            del kwargs["M"]
        except KeyError:
            self.M = 10

        self._init = True
        X = self._init_X(initx, Ylist)
        Z = self._init_Z(initz, X)
        self.bgplvms = [Bayesian_GPLVM(Y, kernel=k(), X=X, Z=Z, M=self.M, **kwargs) for Y in Ylist]
        del self._init

        self.gref = self.bgplvms[0]
        nparams = numpy.array([0] + [sparse_GP._get_params(g).size - g.Z.size for g in self.bgplvms])
        self.nparams = nparams.cumsum()

        self.N = self.gref.N
        self.NQ = self.N * self.Q
        self.MQ = self.M * self.Q

        model.__init__(self)  # @UndefinedVariable

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
    def Ylist(self):
        return [g.likelihood.Y for g in self.bgplvms]
    @Ylist.setter
    def Ylist(self, Ylist):
        for g, Y in itertools.izip(self.bgplvms, Ylist):
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
        self._init_X(initx, self.Ylist)
        self._init_Z(initz, self.X)

    def _get_param_names(self):
        # X_names = sum([['X_%i_%i' % (n, q) for q in range(self.Q)] for n in range(self.N)], [])
        # S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.Q)] for n in range(self.N)], [])
        n1 = self.gref._get_param_names()
        n1var = n1[:self.NQ * 2 + self.MQ]
        map_names = lambda ns, name: map(lambda x: "{1}_{0}".format(*x),
                                         itertools.izip(ns,
                                                        itertools.repeat(name)))
        return list(itertools.chain(n1var, *(map_names(\
                sparse_GP._get_param_names(g)[self.MQ:], n) \
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

        if self._debug:
            for g in self.bgplvms:
                assert numpy.allclose(g.X.ravel(), X)
                assert numpy.allclose(g.X_variance.ravel(), X_var)
                assert numpy.allclose(g.Z.ravel(), Z)

        thetas = [sparse_GP._get_params(g)[g.Z.size:] for g in self.bgplvms]
        params = numpy.hstack([X, X_var, Z, numpy.hstack(thetas)])
        return params

#     def _set_var_params(self, g, X, X_var, Z):
#         g.X = X.reshape(self.N, self.Q)
#         g.X_variance = X_var.reshape(self.N, self.Q)
#         g.Z = Z.reshape(self.M, self.Q)
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

        if self._debug:
            for g in self.bgplvms:
                assert numpy.allclose(g.X, self.gref.X)
                assert numpy.allclose(g.X_variance, self.gref.X_variance)
                assert numpy.allclose(g.Z, self.gref.Z)

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


    def log_likelihood(self):
        ll = -self.gref.KL_divergence()
        for g in self.bgplvms:
            ll += sparse_GP.log_likelihood(g)
        return ll

    def _log_likelihood_gradients(self):
        dLdmu, dLdS = reduce(lambda a, b: [a[0] + b[0], a[1] + b[1]], (g.dL_dmuS() for g in self.bgplvms))
        dKLmu, dKLdS = self.gref.dKL_dmuS()
        dLdmu -= dKLmu
        dLdS -= dKLdS
        dLdmuS = numpy.hstack((dLdmu.flatten(), dLdS.flatten())).flatten()
        dldzt1 = reduce(lambda a, b: a + b, (sparse_GP._log_likelihood_gradients(g)[:self.MQ] for g in self.bgplvms))

        return numpy.hstack((dLdmuS,
                             dldzt1,
                numpy.hstack([numpy.hstack([g.dL_dtheta(),
                                            g.likelihood._gradients(\
                                                partial=g.partial_for_likelihood)]) \
                              for g in self.bgplvms])))

    def _init_X(self, init='PCA', Ylist=None):
        if Ylist is None:
            Ylist = self.Ylist
        if init in "PCA_single":
            X = numpy.zeros((Ylist[0].shape[0], self.Q))
            for qs, Y in itertools.izip(numpy.array_split(numpy.arange(self.Q), len(Ylist)), Ylist):
                X[:, qs] = PCA(Y, len(qs))[0]
        elif init in "PCA_concat":
            X = PCA(numpy.hstack(Ylist), self.Q)[0]
        else:  # init == 'random':
            X = numpy.random.randn(Ylist[0].shape[0], self.Q)
        self.X = X
        return X


    def _init_Z(self, init="permute", X=None):
        if X is None:
            X = self.X
        if init in "permute":
            Z = numpy.random.permutation(X.copy())[:self.M]
        elif init in "random":
            Z = numpy.random.randn(self.M, self.Q) * X.var()
        self.Z = Z
        return Z

    def _handle_plotting(self, fig_num, axes, plotf):
        if axes is None:
            fig = pylab.figure(num=fig_num, figsize=(4 * len(self.bgplvms), 3))
        for i, g in enumerate(self.bgplvms):
            if axes is None:
                ax = fig.add_subplot(1, len(self.bgplvms), i + 1)
            else:
                ax = axes[i]
            plotf(i, g, ax)
        pylab.draw()
        if axes is None:
            fig.tight_layout()
            return fig
        else:
            return pylab.gcf()

    def plot_X_1d(self):
        return self.gref.plot_X_1d()

    def plot_X(self, fig_num="MRD Predictions", axes=None):
        fig = self._handle_plotting(fig_num, axes, lambda i, g, ax: ax.imshow(g.X))
        return fig

    def plot_predict(self, fig_num="MRD Predictions", axes=None):
        fig = self._handle_plotting(fig_num, axes, lambda i, g, ax: ax.imshow(g.predict(g.X)[0]))
        return fig

    def plot_scales(self, fig_num="MRD Scales", axes=None, *args, **kwargs):
        fig = self._handle_plotting(fig_num, axes, lambda i, g, ax: g.kern.plot_ARD(ax=ax, *args, **kwargs))
        return fig

    def plot_latent(self, fig_num="MRD Latent Spaces", axes=None, *args, **kwargs):
        fig = self._handle_plotting(fig_num, axes, lambda i, g, ax: g.plot_latent(ax=ax, *args, **kwargs))
        return fig

    def _debug_plot(self):
        self.plot_X_1d()
        fig = pylab.figure("MRD DEBUG PLOT", figsize=(4 * len(self.bgplvms), 9))
        fig.clf()
        axes = [fig.add_subplot(3, len(self.bgplvms), i + 1) for i in range(len(self.bgplvms))]
        self.plot_X(axes=axes)
        axes = [fig.add_subplot(3, len(self.bgplvms), i + len(self.bgplvms) + 1) for i in range(len(self.bgplvms))]
        self.plot_latent(axes=axes)
        axes = [fig.add_subplot(3, len(self.bgplvms), i + 2 * len(self.bgplvms) + 1) for i in range(len(self.bgplvms))]
        self.plot_scales(axes=axes)
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

