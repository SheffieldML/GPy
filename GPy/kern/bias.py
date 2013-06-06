# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
import hashlib

class bias(Kernpart):
    def __init__(self,input_dim,variance=1.):
        """
        :param input_dim: the number of input dimensions
        :type input_dim: int
        :param variance: the variance of the kernel
        :type variance: float
        """
        self.input_dim = input_dim
        self.num_params = 1
        self.name = 'bias'
        self._set_params(np.array([variance]).flatten())

    def _get_params(self):
        return self.variance

    def _set_params(self,x):
        assert x.shape==(1,)
        self.variance = x

    def _get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        target += self.variance

    def Kdiag(self,X,target):
        target += self.variance

    def dK_dtheta(self,dL_dKdiag,X,X2,target):
        target += dL_dKdiag.sum()

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        target += dL_dKdiag.sum()

    def dK_dX(self, dL_dK,X, X2, target):
        pass

    def dKdiag_dX(self,dL_dKdiag,X,target):
        pass

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, mu, S, target):
        target += self.variance

    def psi1(self, Z, mu, S, target):
        self._psi1 = self.variance
        target += self._psi1
        
    def psi2(self, Z, mu, S, target):
        target += self.variance**2

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S, target):
        target += dL_dpsi0.sum()

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S, target):
        target += dL_dpsi1.sum()

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S, target):
        target += 2.*self.variance*dL_dpsi2.sum()

    def dpsi0_dZ(self, dL_dpsi0, Z, mu, S, target):
        pass

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S, target_mu, target_S):
        pass

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S, target):
        pass

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S, target_mu, target_S):
        pass

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S, target):
        pass

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S, target_mu, target_S):
        pass
