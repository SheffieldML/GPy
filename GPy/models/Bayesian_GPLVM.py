# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys, pdb
from GPLVM import GPLVM
from sparse_GP import sparse_GP
from GPy.util.linalg import pdinv
from ..likelihoods import Gaussian
from .. import kern

class Bayesian_GPLVM(sparse_GP, GPLVM):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, Q, X=None, X_variance=None, init='PCA', M=10, Z=None, kernel=None, **kwargs):
        if X == None:
            X = self.initialise_latent(init, Q, Y)

        if X_variance is None:
            X_variance = np.ones_like(X) * 0.5

        if Z is None:
            Z = np.random.permutation(X.copy())[:M]
        assert Z.shape[1] == X.shape[1]

        if kernel is None:
            kernel = kern.rbf(Q) + kern.white(Q)


        sparse_GP.__init__(self, X, Gaussian(Y), kernel, Z=Z, X_variance=X_variance, **kwargs)

    def _get_param_names(self):
        X_names = sum([['X_%i_%i' % (n, q) for q in range(self.Q)] for n in range(self.N)], [])
        S_names = sum([['X_variance_%i_%i' % (n, q) for q in range(self.Q)] for n in range(self.N)], [])
        return (X_names + S_names + sparse_GP._get_param_names(self))

    def _get_params(self):
        """
        Horizontally stacks the parameters in order to present them to the optimizer.
        The resulting 1-D array has this structure:

        ===============================================================
        |       mu       |        S        |    Z    | theta |  beta  |
        ===============================================================

        """
        return np.hstack((self.X.flatten(), self.X_variance.flatten(), sparse_GP._get_params(self)))

    def _set_params(self, x):
        N, Q = self.N, self.Q
        self.X = x[:self.X.size].reshape(N, Q).copy()
        self.X_variance = x[(N * Q):(2 * N * Q)].reshape(N, Q).copy()
        sparse_GP._set_params(self, x[(2 * N * Q):])


    def dKL_dmuS(self):
        dKL_dS = (1. - (1. / self.X_variance)) * 0.5
        dKL_dmu = self.X
        return dKL_dmu, dKL_dS

    def dL_dmuS(self):
        dL_dmu_psi0, dL_dS_psi0 = self.kern.dpsi1_dmuS(self.dL_dpsi1, self.Z, self.X, self.X_variance)
        dL_dmu_psi1, dL_dS_psi1 = self.kern.dpsi0_dmuS(self.dL_dpsi0, self.Z, self.X, self.X_variance)
        dL_dmu_psi2, dL_dS_psi2 = self.kern.dpsi2_dmuS(self.dL_dpsi2, self.Z, self.X, self.X_variance)
        dL_dmu = dL_dmu_psi0 + dL_dmu_psi1 + dL_dmu_psi2
        dL_dS = dL_dS_psi0 + dL_dS_psi1 + dL_dS_psi2

        return dL_dmu, dL_dS

    def KL_divergence(self):
        var_mean = np.square(self.X).sum()
        var_S = np.sum(self.X_variance - np.log(self.X_variance))
        return 0.5 * (var_mean + var_S) - 0.5 * self.Q * self.N

    def log_likelihood(self):
        return sparse_GP.log_likelihood(self) - self.KL_divergence()

    def _log_likelihood_gradients(self):
        dKL_dmu, dKL_dS = self.dKL_dmuS()
        dL_dmu, dL_dS = self.dL_dmuS()
        # TODO: find way to make faster
        dbound_dmuS = np.hstack(((dL_dmu - dKL_dmu).flatten(), (dL_dS - dKL_dS).flatten()))
        return np.hstack((dbound_dmuS.flatten(), sparse_GP._log_likelihood_gradients(self)))

    def plot_latent(self, which_indices=None, *args, **kwargs):

        if which_indices is None:
            try:
                input_1, input_2 = np.argsort(self.input_sensitivity())[:2]
            except:
                raise ValueError, "cannot Atomatically determine which dimensions to plot, please pass 'which_indices'"
        else:
            input_1, input_2 = which_indices
        ax = GPLVM.plot_latent(self, which_indices=[input_1, input_2], *args, **kwargs)
        ax.plot(self.Z[:, input_1], self.Z[:, input_2], '^w')
        return ax
