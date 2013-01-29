# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys, pdb
from GPLVM import GPLVM
from sparse_GP_regression import sparse_GP_regression
from GPy.util.linalg import pdinv

class Bayesian_GPLVM(sparse_GP_regression, GPLVM):
    """
    Bayesian Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param Q: latent dimensionality
    :type Q: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, Q, init='PCA', **kwargs):
        X = self.initialise_latent(init, Q, Y)
        S = np.ones_like(X) * 1e-2# 
        sparse_GP_regression.__init__(self, X, Y, X_uncertainty = S, **kwargs)

    def get_param_names(self):
        X_names = sum([['X_%i_%i'%(n,q) for n in range(self.N)] for q in range(self.Q)],[])
        S_names = sum([['S_%i_%i'%(n,q) for n in range(self.N)] for q in range(self.Q)],[])
        return (X_names + S_names + sparse_GP_regression.get_param_names(self))

    def get_param(self):
        """
        Horizontally stacks the parameters in order to present them to the optimizer.
        The resulting 1-D array has this structure:

        ===============================================================
        |       mu       |        S        |    Z    | beta |  theta  |
        ===============================================================

        """
        return np.hstack((self.X.flatten(), self.X_uncertainty.flatten(), sparse_GP_regression.get_param(self)))

    def set_param(self,x):
        N, Q = self.N, self.Q
        self.X = x[:self.X.size].reshape(N,Q).copy()
        self.X_uncertainty = x[(N*Q):(2*N*Q)].reshape(N,Q).copy()
        sparse_GP_regression.set_param(self, x[(2*N*Q):])

    def dL_dmuS(self):
        dL_dmu_psi0, dL_dS_psi0 = self.kern.dpsi1_dmuS(self.dL_dpsi1,self.Z,self.X,self.X_uncertainty)
        dL_dmu_psi1, dL_dS_psi1 = self.kern.dpsi0_dmuS(self.dL_dpsi0,self.Z,self.X,self.X_uncertainty)
        dL_dmu_psi2, dL_dS_psi2 = self.kern.dpsi2_dmuS(self.dL_dpsi2,self.Z,self.X,self.X_uncertainty)
        dL_dmu = dL_dmu_psi0 + dL_dmu_psi1 + dL_dmu_psi2
        dL_dS = dL_dS_psi0 + dL_dS_psi1 + dL_dS_psi2

        return np.hstack((dL_dmu.flatten(), dL_dS.flatten()))

    def log_likelihood_gradients(self):
        return np.hstack((self.dL_dmuS().flatten(), sparse_GP_regression.log_likelihood_gradients(self)))

