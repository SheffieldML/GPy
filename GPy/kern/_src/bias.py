# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
import numpy as np

class Bias(Kern):
    def __init__(self,input_dim,variance=1.,name=None):
        super(Bias, self).__init__(input_dim, name)
        self.variance = Param("variance", variance, Logexp())
        self.add_parameter(self.variance)

    def K(self, X, X2=None):
        shape = (X.shape[0], X.shape[0] if X2 is None else X2.shape[0])
        ret = np.empty(shape, dtype=np.float64)
        ret[:] = self.variance
        return ret

    def Kdiag(self,X):
        ret = np.empty((X.shape[0],), dtype=np.float64)
        ret[:] = self.variance
        return ret

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = dL_dK.sum()

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = dL_dK.sum()

    def gradients_X(self, dL_dK,X, X2, target):
        return np.zeros(X.shape)

    def gradients_X_diag(self,dL_dKdiag,X,target):
        return np.zeros(X.shape)


    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, mu, S):
        return self.Kdiag(mu)

    def psi1(self, Z, mu, S, target):
        return self.K(mu, S)

    def psi2(self, Z, mu, S, target):
        ret = np.empty((mu.shape[0], Z.shape[0], Z.shape[0]), dtype=np.float64)
        ret[:] = self.variance**2
        return ret

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        self.variance.gradient = dL_dKmm.sum() + dL_dpsi0.sum() + dL_dpsi1.sum() + 2.*self.variance*dL_dpsi2.sum()

    def gradients_Z_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        return np.zeros(Z.shape)

    def gradients_muS_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        return np.zeros(mu.shape), np.zeros(S.shape)
