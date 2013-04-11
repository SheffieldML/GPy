'''
Created on 10 Apr 2013

@author: Max Zwiessele
'''
from GPy.core import model
from GPy.models.Bayesian_GPLVM import Bayesian_GPLVM
import numpy
from GPy.models.sparse_GP import sparse_GP
import itertools
from matplotlib import pyplot
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
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'
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
        self._debug = False
        if kwargs.has_key("_debug"):
            self._debug = kwargs['_debug']
            del kwargs['_debug']
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
        self.bgplvms = [Bayesian_GPLVM(Y, kernel=k(), **kwargs) for Y in Ylist]
        self.gref = self.bgplvms[0]
        nparams = numpy.array([0] + [sparse_GP._get_params(g).size - g.Z.size for g in self.bgplvms])
        self.nparams = nparams.cumsum()
        self.Q = self.gref.Q
        self.N = self.gref.N
        self.NQ = self.N * self.Q
        self.M = self.gref.M
        self.MQ = self.M * self.Q

        model.__init__(self)  # @UndefinedVariable

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
        X = self.gref.X.flatten()
        X_var = self.gref.X_variance.flatten()
        Z = self.gref.Z.flatten()
        thetas = [sparse_GP._get_params(g)[g.Z.size:] for g in self.bgplvms]
        params = numpy.hstack([X, X_var, Z, numpy.hstack(thetas)])
        return params

    def _set_var_params(self, g, X, X_var, Z):
        g.X = X
        g.X_variance = X_var
        g.Z = Z

    def _set_kern_params(self, g, p):
        g.kern._set_params(p[:g.kern.Nparam])
        g.likelihood._set_params(p[g.kern.Nparam:])

    def _set_params(self, x):
        start = 0; end = self.NQ
        X = x[start:end].reshape(self.N, self.Q).copy()
        start = end; end += start
        X_var = x[start:end].reshape(self.N, self.Q).copy()
        start = end; end += self.MQ
        Z = x[start:end].reshape(self.M, self.Q).copy()
        thetas = x[end:]

        # set params for all others:
        for g, s, e in itertools.izip(self.bgplvms, self.nparams, self.nparams[1:]):
            self._set_var_params(g, X, X_var, Z)
            self._set_kern_params(g, thetas[s:e].copy())
            g._compute_kernel_matrices()
            g._computations()


    def log_likelihood(self):
        ll = self.gref.KL_divergence()
        for g in self.bgplvms:
            ll += sparse_GP.log_likelihood(g)
        return ll

    def _log_likelihood_gradients(self):
        dldmus = self.gref.dL_dmuS().flatten()
        dldzt1 = sparse_GP._log_likelihood_gradients(self.gref)
        return numpy.hstack((dldmus,
                             dldzt1,
                numpy.hstack([numpy.hstack([g.dL_dtheta(),
                                            g.likelihood._gradients(\
                                                partial=g.partial_for_likelihood)]) \
                              for g in self.bgplvms[1:]])))

    def plot_scales(self):
        fig = pylab.figure("MRD Scales", figsize=(4 * len(self.bgplvms), 3))
        for i, g in enumerate(self.bgplvms):
            ax = fig.add_subplot(1, len(self.bgplvms), i + 1)
            g.kern.plot_ARD(ax=ax)
