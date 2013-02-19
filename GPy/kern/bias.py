# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib

class bias(kernpart):
    def __init__(self,D,variance=1.):
        """
        :param D: the number of input dimensions
        :type D: int
        :param variance: the variance of the kernel
        :type variance: float
        """
        self.D = D
        self.Nparam = 1
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

    def dK_dtheta(self,partial,X,X2,target):
        target += partial.sum()

    def dKdiag_dtheta(self,partial,X,target):
        target += partial.sum()

    def dK_dX(self, partial,X, X2, target):
        pass

    def dKdiag_dX(self,partial,X,target):
        pass

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, mu, S, target):
        target += self.variance

    def psi1(self, Z, mu, S, target):
        target += self.variance

    def psi2(self, Z, mu, S, target):
        target += self.variance**2

    def dpsi0_dtheta(self, partial, Z, mu, S, target):
        target += partial.sum()

    def dpsi1_dtheta(self, partial, Z, mu, S, target):
        target += partial.sum()

    def dpsi2_dtheta(self, partial, Z, mu, S, target):
        target += 2.*self.variance*partial.sum()

    
    def dpsi0_dZ(self, partial, Z, mu, S, target):
        pass

    def dpsi0_dmuS(self, partial, Z, mu, S, target_mu, target_S):
        pass

    def dpsi1_dZ(self, partial, Z, mu, S, target):
        pass

    def dpsi1_dmuS(self, partial, Z, mu, S, target_mu, target_S):
        pass

    def dpsi2_dZ(self, partial, Z, mu, S, target):
        pass

    def dpsi2_dmuS(self, partial, Z, mu, S, target_mu, target_S):
        pass
