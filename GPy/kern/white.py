# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
class white(kernpart):
    """
    White noise kernel.

    :param D: the number of input dimensions
    :type D: int
    :param variance:
    :type variance: float
    """
    def __init__(self,D,variance=1.):
        self.D = D
        self.Nparam = 1
        self.name = 'white'
        self.set_param(np.array([variance]).flatten())

    def get_param(self):
        return self.variance

    def set_param(self,x):
        assert x.shape==(1,)
        self.variance = x

    def get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        if X.shape==X2.shape:
            if np.all(X==X2):
                np.add(target,np.eye(X.shape[0])*self.variance,target)

    def Kdiag(self,X,target):
        target += self.variance

    def dK_dtheta(self,partial,X,X2,target):
        if X.shape==X2.shape:
            if np.all(X==X2):
                target += np.trace(partial)

    def dKdiag_dtheta(self,X,target):
        np.add(target[:,0],1.,target[:,0])

    def dK_dX(self,partial,X,X2,target):
        pass

    def dKdiag_dX(self,X,target):
        pass

    def psi0(self,Z,mu,S,target):
        target += self.variance

    def dpsi0_dtheta(self,Z,mu,S,target):
        target += 1.

    def dpsi0_dmuS(self,Z,mu,S,target_mu,target_S):
        pass

    def psi1(self,Z,mu,S,target):
        pass

    def dpsi1_dtheta(self,Z,mu,S,target):
        pass

    def dpsi1_dZ(self,Z,mu,S,target):
        pass

    def dpsi1_dmuS(self,Z,mu,S,target_mu,target_S):
        pass

    def psi2(self,Z,mu,S,target):
        pass

    def dpsi2_dZ(self,Z,mu,S,target):
        pass

    def dpsi2_dtheta(self,Z,mu,S,target):
        pass

    def dpsi2_dmuS(self,Z,mu,S,target_mu,target_S):
        pass

