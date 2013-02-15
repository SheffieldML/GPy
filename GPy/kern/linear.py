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
    :type variances: array or list of the appropriate size (or float if there is only one variance parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel has only one variance parameter \sigma^2, otherwise there is one variance parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object
    """

    def __init__(self,D,variances=None,ARD=False):
        self.D = D
        self.ARD = ARD
        if ARD == False:
            self.Nparam = 1
            self.name = 'linear'
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == 1, "Only one variance needed for non-ARD kernel"
            else:
                variances = np.ones(1)
            self._Xcache, self._X2cache = np.empty(shape=(2,))
        else:
            self.Nparam = self.D
            self.name = 'linear'
            if variances is not None:
                variances = np.asarray(variances)
                assert variances.size == self.D, "bad number of lengthscales"
            else:
                variances = np.ones(self.D)
        self._set_params(variances.flatten())

        #initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3,1))
        self._X, self._X2, self._params = np.empty(shape=(3,1))

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
        self._psi_computations(Z,mu,S)
        target += np.sum(self.variances*self.mu2_S,1)

    def dpsi0_dtheta(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        tmp = partial[:, None] * self.mu2_S
        if self.ARD:
            target += tmp.sum(0)
        else:
            target += tmp.sum()

    def dpsi0_dmuS(self,partial, Z,mu,S,target_mu,target_S):
        target_mu += partial[:, None] * (2.0*mu*self.variances)
        target_S += partial[:, None] * self.variances

    def psi1(self,Z,mu,S,target):
        """the variance, it does nothing"""
        self.K(mu,Z,target)

    def dpsi1_dtheta(self,partial,Z,mu,S,target):
        """the variance, it does nothing"""
        self.dK_dtheta(partial,mu,Z,target)

    def dpsi1_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """Do nothing for S, it does not affect psi1"""
        self._psi_computations(Z,mu,S)
        target_mu += (partial.T[:,:, None]*(Z*self.variances)).sum(1)

    def dpsi1_dZ(self,partial,Z,mu,S,target):
        self.dK_dX(partial.T,Z,mu,target)

    def psi2(self,Z,mu,S,target):
        """
        returns N,M,M matrix
        """
        self._psi_computations(Z,mu,S)
        psi2 = self.ZZ*np.square(self.variances)*self.mu2_S[:, None, None, :]
        target += psi2.sum(-1)

    def dpsi2_dtheta(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        tmp = (partial[:,:,:,None]*(2.*self.ZZ*self.mu2_S[:,None,None,:]*self.variances))
        if self.ARD:
            target += tmp.sum(0).sum(0).sum(0)
        else:
            target += tmp.sum()

    def dpsi2_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """Think N,M,M,Q """
        self._psi_computations(Z,mu,S)
        tmp = self.ZZ*np.square(self.variances) # M,M,Q
        target_mu += (partial[:,:,:,None]*tmp*2.*mu[:,None,None,:]).sum(1).sum(1)
        target_S += (partial[:,:,:,None]*tmp).sum(1).sum(1)

    def dpsi2_dZ(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        mu2_S = np.sum(self.mu2_S,0)# Q,
        target += (partial[:,:,:,None] * (self.mu2_S[:,None,None,:]*(Z*np.square(self.variances)[None,:])[None,None,:,:])).sum(0).sum(1)

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

    def _psi_computations(self,Z,mu,S):
        #here are the "statistics" for psi1 and psi2
        if not np.all(Z==self._Z):
            #Z has changed, compute Z specific stuff
            self.ZZ = Z[:,None,:]*Z[None,:,:] # M,M,Q
            self._Z = Z
        if not (np.all(mu==self._mu) and np.all(S==self._S)):
            self.mu2_S = np.square(mu)+S
            self._mu, self._S = mu, S
