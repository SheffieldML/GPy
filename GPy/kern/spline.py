# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib
def theta(x):
    """Heaviside step function"""
    return np.where(x>=0.,1.,0.)

class spline(kernpart):
    """
    Spline kernel

    :param D: the number of input dimensions (fixed to 1 right now TODO)
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float

    """

    def __init__(self,D,variance=1.,lengthscale=1.):
        self.D = D
        assert self.D==1
        self.Nparam = 1
        self.name = 'spline'
        self._set_params(np.squeeze(variance))

    def _get_params(self):
        return self.variance

    def _set_params(self,x):
        self.variance = x

    def _get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        assert np.all(X>0), "Spline covariance is for +ve domain only. TODO: symmetrise"
        assert np.all(X2>0), "Spline covariance is for +ve domain only. TODO: symmetrise"
        t = X
        s = X2.T
        s_t = s-t # broadcasted subtraction
        target += self.variance*(0.5*(t*s**2) - s**3/6. + (s_t)**3*theta(s_t)/6.)

    def Kdiag(self,X,target):
        target += self.variance*X.flatten()**3/3.

    def dK_dtheta(self,X,X2,target):
        target += 0.5*(t*s**2) - s**3/6. + (s_t)**3*theta(s_t)/6.

    def dKdiag_dtheta(self,X,target):
        target += X.flatten()**3/3.

    def dKdiag_dX(self,X,target):
        target += self.variance*X**2

