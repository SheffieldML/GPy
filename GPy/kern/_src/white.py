# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kern import Kern
import numpy as np
from ..core.parameterization import Param
from ..core.parameterization.transformations import Logexp

class White(Kern):
    """
    White noise kernel.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """
    def __init__(self,input_dim,variance=1., name='white'):
        super(White, self).__init__(input_dim, name)
        self.input_dim = input_dim
        self.variance = Param('variance', variance, Logexp())
        self.add_parameters(self.variance)
        self._psi1 = 0 # TODO: more elegance here

    def K(self,X,X2):
        if X2 is None:
            return np.eye(X.shape[0])*self.variance

    def Kdiag(self,X):
        ret = np.ones(X.shape[0])
        ret[:] = self.variance
        return ret

    def update_gradients_full(self, dL_dK, X):
        self.variance.gradient = np.trace(dL_dK)

    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        self.variance.gradient = np.trace(dL_dKmm) + np.sum(dL_dKdiag)

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        raise NotImplementedError

    def gradients_X(self,dL_dK,X,X2):
        return np.zeros_like(X)

    def psi0(self,Z,mu,S,target):
        pass # target += self.variance

    def dpsi0_dtheta(self,dL_dpsi0,Z,mu,S,target):
        pass # target += dL_dpsi0.sum()

    def dpsi0_dmuS(self,dL_dpsi0,Z,mu,S,target_mu,target_S):
        pass

    def psi1(self,Z,mu,S,target):
        pass

    def dpsi1_dtheta(self,dL_dpsi1,Z,mu,S,target):
        pass

    def dpsi1_dZ(self,dL_dpsi1,Z,mu,S,target):
        pass

    def dpsi1_dmuS(self,dL_dpsi1,Z,mu,S,target_mu,target_S):
        pass

    def psi2(self,Z,mu,S,target):
        pass

    def dpsi2_dZ(self,dL_dpsi2,Z,mu,S,target):
        pass

    def dpsi2_dtheta(self,dL_dpsi2,Z,mu,S,target):
        pass

    def dpsi2_dmuS(self,dL_dpsi2,Z,mu,S,target_mu,target_S):
        pass
