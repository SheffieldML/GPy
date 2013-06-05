# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np

def theta(x):
    """Heavisdie step function"""
    return np.where(x>=0.,1.,0.)

class Brownian(Kernpart):
    """
    Brownian Motion kernel.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance:
    :type variance: float
    """
    def __init__(self,input_dim,variance=1.):
        self.input_dim = input_dim
        assert self.input_dim==1, "Brownian motion in 1D only"
        self.num_params = 1.
        self.name = 'Brownian'
        self._set_params(np.array([variance]).flatten())

    def _get_params(self):
        return self.variance

    def _set_params(self,x):
        assert x.shape==(1,)
        self.variance = x

    def _get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        if X2 is None:
            X2 = X
        target += self.variance*np.fmin(X,X2.T)

    def Kdiag(self,X,target):
        target += self.variance*X.flatten()

    def dK_dtheta(self,dL_dK,X,X2,target):
        if X2 is None:
            X2 = X
        target += np.sum(np.fmin(X,X2.T)*dL_dK)

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        target += np.dot(X.flatten(), dL_dKdiag)

    def dK_dX(self,dL_dK,X,X2,target):
        raise NotImplementedError, "TODO"
        #target += self.variance
        #target -= self.variance*theta(X-X2.T)
        #if X.shape==X2.shape:
            #if np.all(X==X2):
                #np.add(target[:,:,0],self.variance*np.diag(X2.flatten()-X.flatten()),target[:,:,0])


    def dKdiag_dX(self,dL_dKdiag,X,target):
        target += self.variance*dL_dKdiag[:,None]

