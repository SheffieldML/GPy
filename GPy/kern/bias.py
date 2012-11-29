# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib

class bias(kernpart):
    def __init__(self,D,variance=1.):
        """
        Arguments
        ----------
        D: int - the number of input dimensions
        variance: float
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
        if X2 is None: X2 = X
        np.add(self.variance, target,target)

    def Kdiag(self,X,target):
        np.add(target,self.variance,target)

    def dK_dtheta(self,X,X2,target):
        """Return shape is NxMx(Ntheta)"""
        if X2 is None: X2 = X
        np.add(target[:,:,0],1., target[:,:,0])

    def dKdiag_dtheta(self,X,target):
        np.add(target[:,0],1.,target[:,0])

    def dK_dX(self, X, X2, target):
        pass

    def dKdiag_dX(self,X,target):
        pass
