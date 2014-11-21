# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
from ...core.parameterization import Param

def theta(x):
    """Heaviside step function"""
    return np.where(x>=0.,1.,0.)

class Spline(Kernpart):
    """
    Spline kernel

    :param input_dim: the number of input dimensions (fixed to 1 right now TODO)
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float

    """

    def __init__(self,input_dim,variance=1.,lengthscale=1.):
        self.input_dim = input_dim
        assert self.input_dim==1
        self.num_params = 1
        self.name = 'spline'
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengthscale', lengthscale)
        self.add_parameters(self.variance, self.lengthscale)
        
#     def _get_params(self):
#         return self.variance
# 
#     def _set_params(self,x):
#         self.variance = x
# 
#     def _get_param_names(self):
#         return ['variance']

    def K(self,X,X2,target):
        assert np.all(X>0), "Spline covariance is for +ve domain only. TODO: symmetrise"
        assert np.all(X2>0), "Spline covariance is for +ve domain only. TODO: symmetrise"
        t = X
        s = X2.T
        s_t = s-t # broadcasted subtraction
        target += self.variance*(0.5*(t*s**2) - s**3/6. + (s_t)**3*theta(s_t)/6.)

    def Kdiag(self,X,target):
        target += self.variance*X.flatten()**3/3.

    def _param_grad_helper(self,X,X2,target):
        target += 0.5*(t*s**2) - s**3/6. + (s_t)**3*theta(s_t)/6.

    def dKdiag_dtheta(self,X,target):
        target += X.flatten()**3/3.

    def dKdiag_dX(self,X,target):
        target += self.variance*X**2

