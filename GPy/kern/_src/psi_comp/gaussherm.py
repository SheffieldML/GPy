# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
An approximated psi-statistics implementation based on Gauss-Hermite Quadrature
"""

import numpy as np

from ....core.parameterization import Param
from GPy.util.caching import Cache_this
from ....util.linalg import tdot
from . import PSICOMP

class PSICOMP_GH(PSICOMP):
    """
    TODO: support Psi2 with shape NxMxM
    """
    
    def __init__(self, degree=5, cache_K=True):
        self.degree = degree
        self.cache_K = cache_K
        self.locs, self.weights = np.polynomial.hermite.hermgauss(degree)
        self.locs *= np.sqrt(2.)
        self.weights*= 1./np.sqrt(np.pi)
        self.Xs = None

    def _setup_observers(self):
        pass
    
    @Cache_this(limit=10, ignore_args=(0,))
    def comp_K(self, Z, qX):
        if self.Xs is None or self.Xs.shape != qX.mean.shape:
            from ....core.parameterization import ObsAr
            self.Xs = ObsAr(np.empty((self.degree,)+qX.mean.shape))
        mu, S = qX.mean.values, qX.variance.values
        S_sq = np.sqrt(S)
        for i in xrange(self.degree):
            self.Xs[i] = self.locs[i]*S_sq+mu
        return self.Xs
    
    @Cache_this(limit=10, ignore_args=(0,))
    def psicomputations(self, kern, Z, qX, return_psi2_n=False):
        mu, S = qX.mean.values, qX.variance.values
        N,M,Q = mu.shape[0],Z.shape[0],mu.shape[1]
        if self.cache_K: Xs = self.comp_K(Z, qX)
        else: S_sq = np.sqrt(S)
        
        psi0 = np.zeros((N,))
        psi1 = np.zeros((N,M))
        psi2 = np.zeros((M,M))
        for i in xrange(self.degree):
            if self.cache_K:
                X = Xs[i]
            else:
                X = self.locs[i]*S_sq+mu
            psi0 += self.weights[i]* kern.Kdiag(X)
            Kfu = kern.K(X,Z)
            psi1 += self.weights[i]* Kfu
            psi2 += self.weights[i]* tdot(Kfu.T)
        return psi0, psi1, psi2
    
    @Cache_this(limit=10, ignore_args=(0, 2,3,4))
    def psiDerivativecomputations(self, kern, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, qX):
        mu, S = qX.mean.values, qX.variance.values
        if self.cache_K: Xs = self.comp_K(Z, qX)
        S_sq = np.sqrt(S)
        
        dtheta_old = kern.gradient.copy()
        dtheta = np.zeros_like(kern.gradient)
        if isinstance(Z, Param):
            dZ = np.zeros_like(Z.values)
        else:
            dZ = np.zeros_like(Z)
        dmu = np.zeros_like(mu)
        dS = np.zeros_like(S)
        for i in xrange(self.degree):
            if self.cache_K:
                X = Xs[i]
            else:
                X = self.locs[i]*S_sq+mu
            dL_dpsi0_i = dL_dpsi0*self.weights[i]
            kern.update_gradients_diag(dL_dpsi0_i, X)
            dtheta += kern.gradient
            dX = kern.gradients_X_diag(dL_dpsi0_i, X)
            Kfu = kern.K(X,Z)
            dL_dkfu = (dL_dpsi1+ 2.*Kfu.dot(dL_dpsi2))*self.weights[i]
            kern.update_gradients_full(dL_dkfu, X, Z)
            dtheta += kern.gradient
            dX_i, dZ_i = kern.gradients_X_X2(dL_dkfu, X, Z)
            dX += dX_i
            dZ += dZ_i
            dmu += dX
            dS += dX*self.locs[i]/(2.*S_sq)
        kern.gradient[:] = dtheta_old
        return dtheta, dZ, dmu, dS
        



