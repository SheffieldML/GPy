# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
class white(Kernpart):
    """
    White noise kernel.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """
    def __init__(self,input_dim,variance=1.):
        self.input_dim = input_dim
        self.num_params = 1
        self.name = 'white'
        self._set_params(np.array([variance]).flatten())
        self._psi1 = 0 # TODO: more elegance here
        
    def _get_params(self):
        return self.variance

    def _set_params(self,x):
        assert x.shape==(1,)
        self.variance = x

    def _get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        if X2 is None:
            target += np.eye(X.shape[0])*self.variance

    def Kdiag(self,X,target):
        target += self.variance

    def dK_dtheta(self,dL_dK,X,X2,target):
        if X2 is None:
            target += np.trace(dL_dK)

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        target += np.sum(dL_dKdiag)

    def dK_dX(self,dL_dK,X,X2,target):
        pass

    def dKdiag_dX(self,dL_dKdiag,X,target):
        pass

    def psi0(self,Z,mu,S,target):
        target += self.variance

    def dpsi0_dtheta(self,dL_dpsi0,Z,mu,S,target):
        target += dL_dpsi0.sum()

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
