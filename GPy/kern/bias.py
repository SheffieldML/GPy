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
        self.set_param(np.array([variance]).flatten())

    def get_param(self):
        return self.variance

    def set_param(self,x):
        assert x.shape==(1,)
        self.variance = x

    def get_param_names(self):
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
