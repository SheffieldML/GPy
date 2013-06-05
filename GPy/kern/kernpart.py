# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


class Kernpart(object):
    def __init__(self,input_dim):
        """
        The base class for a kernpart: a positive definite function which forms part of a kernel

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        self.input_dim = input_dim
        self.num_params = 1
        self.name = 'unnamed'

    def _get_params(self):
        raise NotImplementedError
    def _set_params(self,x):
        raise NotImplementedError
    def _get_param_names(self):
        raise NotImplementedError
    def K(self,X,X2,target):
        raise NotImplementedError
    def Kdiag(self,X,target):
        raise NotImplementedError
    def dK_dtheta(self,dL_dK,X,X2,target):
        raise NotImplementedError
    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        raise NotImplementedError
    def psi0(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dtheta(self,dL_dpsi0,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dmuS(self,dL_dpsi0,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi1(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dtheta(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dZ(self,dL_dpsi1,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dmuS(self,dL_dpsi1,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi2(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dZ(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dtheta(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dmuS(self,dL_dpsi2,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def dK_dX(self,X,X2,target):
        raise NotImplementedError
