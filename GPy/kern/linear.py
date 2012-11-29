# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
class linear(kernpart):
    """
    Linear kernel

    :param D: the number of input dimensions
    :type D: int
    :param variance: variance
    :type variance: None|float
    """

    def __init__(self, D, variance=None):
        self.D = D
        if variance is None:
            variance = 1.0
        self.Nparam = 1
        self.name = 'linear'
        self.set_param(variance)

    def get_param(self):
        return self.variance

    def set_param(self,x):
        self.variance = x

    def get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        target += self.variance * np.dot(X, X2.T)

    def Kdiag(self,X,target):
        np.add(target,np.sum(self.variance*np.square(X),-1),target)

    def dK_dtheta(self,X,X2,target):
        """
        Computes the derivatives wrt theta
        Return shape is NxMx(Ntheta)

        """

        if X2 is None: X2 = X
        product = np.dot(X, X2.T)[:,:, None]#X[:,None,:]*X2[None,:,:]
        target += product

    def dK_dX(self,X,X2,target):
        if X2 is None: X2 = X
        np.add(target,X2[:,None,:]*self.variance,target)

    # def psi0(self,Z,mu,S,target):
    #     expected = np.square(mu) + S
    #     np.add(target,np.sum(self.variance*expected),target)

    # def dpsi0_dtheta(self,Z,mu,S,target):
    #     expected = np.square(mu) + S
    #     return -2.*np.sum(expected,0)

    # def dpsi0_dmuS(self,Z,mu,S,target_mu,target_S):
    #     np.add(target_mu,2*mu*self.variances,target_mu)
    #     np.add(target_S,self.variances,target_S)

    # def dpsi0_dZ(self,Z,mu,S,target):
    #     pass

    # def psi1(self,Z,mu,S,target):
    #     """the variance, it does nothing"""
    #     self.K(mu,Z,target)

    # def dpsi1_dtheta(self,Z,mu,S,target):
    #     """the variance, it does nothing"""
    #     self.dK_dtheta(mu,Z,target)

    # def dpsi1_dmuS(self,Z,mu,S,target_mu,target_S):
    #     """Do nothing for S, it does not affect psi1"""
    #     np.add(target_mu,Z/self.variances2,target_mu)

    # def dpsi1_dZ(self,Z,mu,S,target):
    #     self.dK_dX(mu,Z,target)

    # def psi2(self,Z,mu,S,target):
    #     """Think N,M,M,Q """
    #     mu2_S = np.square(mu)+SN,Q,
    #     ZZ = Z[:,None,:]*Z[None,:,:] M,M,Q
    #     psi2 = ZZ*np.square(self.variances)*mu2_S
    #     np.add(target, psi2.sum(-1),target) M,M

    # def dpsi2_dtheta(self,Z,mu,S,target):
    #     mu2_S = np.square(mu)+SN,Q,
    #     ZZ = Z[:,None,:]*Z[None,:,:] M,M,Q
    #     target += 2.*ZZ*mu2_S*self.variances

    # def dpsi2_dmuS(self,Z,mu,S,target_mu,target_S):
    #     """Think N,M,M,Q """
    #     mu2_S = np.sum(np.square(mu)+S,0)Q,
    #     ZZ = Z[:,None,:]*Z[None,:,:] M,M,Q
    #     tmp = ZZ*np.square(self.variances) M,M,Q
    #     np.add(target_mu, tmp*2.*mu[:,None,None,:],target_mu) N,M,M,Q
    #     np.add(target_S, tmp, target_S) N,M,M,Q

    # def dpsi2_dZ(self,Z,mu,S,target):
    #     mu2_S = np.sum(np.square(mu)+S,0)Q,
    #     target += Z[:,None,:]*np.square(self.variances)*mu2_S
