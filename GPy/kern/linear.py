# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import kernpart
import numpy as np

class linear(kernpart):
    """
    Linear kernel

    .. math::

       k(x,y) = \sum_{i=1}^D \sigma^2_i x_iy_i

    :param D: the number of input dimensions
    :type D: int
    :param variances: the vector of variances :math:`\sigma^2_i`
    :type variances: np.ndarray of size (1,) or (D,) depending on ARD
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one single variance parameter \sigma^2), otherwise there is one variance parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object
    """

    def __init__(self,D,variances=None,ARD=True):
        self.D = D
        self.ARD = ARD
        if ARD == False:
            self.Nparam = 1
            self.name = 'linear'
            if variances is not None:
                if isinstance(variances, float):
                    variances = np.array([variances])
                    
                assert variances.shape == (1,)
            else:
                variances = np.ones(1)
            self._Xcache, self._X2cache = np.empty(shape=(2,))
        else:
            self.Nparam = self.D
            self.name = 'linear_ARD'
            if variances is not None:
                assert variances.shape == (self.D,)
            else:
                variances = np.ones(self.D)
        self._set_params(variances)

    def _get_params(self):
        return self.variances

    def _set_params(self,x):
        assert x.size==(self.Nparam)
        self.variances = x
        self.variances2 = np.square(self.variances)

    def _get_param_names(self):
        if self.Nparam == 1:
            return ['variance']
        else:
            return ['variance_%i'%i for i in range(self.variances.size)]

    def K(self,X,X2,target):
        if self.ARD:
            XX = X*np.sqrt(self.variances)
            XX2 = X2*np.sqrt(self.variances)
            target += np.dot(XX, XX2.T)
        else:
            self._K_computations(X, X2)
            target += self.variances * self._dot_product

    def Kdiag(self,X,target):
        np.add(target,np.sum(self.variances*np.square(X),-1),target)

    def dK_dtheta(self,partial,X,X2,target):
        if self.ARD:
            product = X[:,None,:]*X2[None,:,:]
            target += (partial[:,:,None]*product).sum(0).sum(0)
        else:
            self._K_computations(X, X2)
            target += np.sum(self._dot_product*partial)

    def dK_dX(self,partial,X,X2,target):
        target += (((X2[:, None, :] * self.variances)) * partial[:,:, None]).sum(0)

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self,Z,mu,S,target):
        expected = np.square(mu) + S
        target += np.sum(self.variances*expected)

    def dpsi0_dtheta(self,partial,Z,mu,S,target):
        expected = np.square(mu) + S
        target += (partial[:, None] * (-2.*np.sum(expected,0))).sum()

    def dpsi0_dmuS(self,partial, Z,mu,S,target_mu,target_S):
        target_mu += partial[:, None] * (2*mu*self.variances)
        target_S += partial[:, None] * self.variances

    def dpsi0_dZ(self,Z,mu,S,target):
        pass

    def psi1(self,Z,mu,S,target):
        """the variance, it does nothing"""
        self.K(mu,Z,target)

    def dpsi1_dtheta(self,partial,Z,mu,S,target):
        """the variance, it does nothing"""
        self.dK_dtheta(partial,mu,Z,target)

    def dpsi1_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """Do nothing for S, it does not affect psi1"""
        target_mu += (partial.T[:,:, None]*(Z/self.variances)).sum(1)

    def dpsi1_dZ(self,partial,Z,mu,S,target):
        self.dK_dX(partial.T,Z,mu,target)

    def psi2(self,Z,mu,S,target):
        """
        returns N,M,M matrix
        """
        mu2_S = np.square(mu)+S# N,Q,
        ZZ = Z[:,None,:]*Z[None,:,:] # M,M,Q
        psi2 = ZZ*np.square(self.variances)*mu2_S[:, None, None, :]
        target += psi2.sum(-1)

    def dpsi2_dtheta(self,partial,Z,mu,S,target):
        mu2_S = np.square(mu)+S# N,Q,
        ZZ = Z[:,None,:]*Z[None,:,:] # M,M,Q
        target += (partial[:,:,:,None]*(2.*ZZ*mu2_S[:,None,None,:]*self.variances)).sum()

    def dpsi2_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """Think N,M,M,Q """
        mu2_S = np.sum(np.square(mu)+S,0)# Q,
        ZZ = Z[:,None,:]*Z[None,:,:] # M,M,Q
        tmp = ZZ*np.square(self.variances) # M,M,Q
        target_mu += (partial[:,:,:,None]*tmp*2.*mu[:,None,None,:]).sum(1).sum(1)
        target_S += (partial[:,:,:,None]*tmp).sum(1).sum(1)

    def dpsi2_dZ(self,partial,Z,mu,S,target):
        mu2_S = np.sum(np.square(mu)+S,0)# Q,
        target += (partial[:,:,:,None]* (Z * mu2_S * np.square(self.variances))).sum(0).sum(0)

    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self,X,X2):
        if X2 is None:
            X2 = X
        if not (np.all(X==self._Xcache) and np.all(X2==self._X2cache)):
            self._Xcache = X
            self._X2cache = X2
            self._dot_product = np.dot(X,X2.T)
        else:
            # print "Cache hit!"
            pass # TODO: insert debug message here (logging framework)
