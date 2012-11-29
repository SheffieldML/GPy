# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib
from scipy import integrate

class exponential(kernpart):
    """
    Exponential kernel (aka Ornstein-Uhlenbeck or Matern 1/2)

    .. math::

       k(r) = \sigma^2 \exp(- r) \qquad \qquad \\text{ where  } r = \sqrt{\sum_{i=1}^D \\frac{(x_i-y_i)^2}{\ell_i^2} }

    :param D: the number of input dimensions
    :type D: int
    :param variance: the variance :math:`\sigma^2`
    :type variance: float
    :param lengthscale: the lengthscales :math:`\ell_i`
    :type lengthscale: np.ndarray of size (D,)
    :rtype: kernel object

    """
    def __init__(self,D,variance=1.,lengthscales=None):
        self.D = D
        if lengthscales is not None:
            assert lengthscales.shape==(self.D,)
        else:
            lengthscales = np.ones(self.D)
        self.Nparam = self.D + 1
        self.name = 'exp'
        self.set_param(np.hstack((variance,lengthscales)))

    def get_param(self):
        """return the value of the parameters."""
        return np.hstack((self.variance,self.lengthscales))
    def set_param(self,x):
        """set the value of the parameters."""
        assert x.size==(self.D+1)
        self.variance = x[0]
        self.lengthscales = x[1:]
    def get_param_names(self):
        """return parameter names."""
        if self.D==1:
            return ['variance','lengthscale']
        else:
            return ['variance']+['lengthscale_%i'%i for i in range(self.lengthscales.size)]
    def K(self,X,X2,target):
        """Compute the covariance matrix between X and X2."""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:,None,:]-X2[None,:,:])/self.lengthscales),-1))
        np.add(self.variance*np.exp(-dist), target,target)
    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        np.add(target,self.variance,target)

    def Gram_matrix(self,F,F1,lower,upper):
        """
        Return the Gram matrix of the vector of functions F with respect to the RKHS norm. The use of this function is limited to D=1.

        :param F: vector of functions
        :type F: np.array
        :param F1: vector of derivatives of F
        :type F1: np.array  
        :param lower,upper: boundaries of the input domain
        :type lower,upper: floats  
        """
        assert self.D == 1
        def L(x,i):
            return(1./self.lengthscales*F[i](x) + F1[i](x))
        n = F.shape[0]
        G = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                G[i,j] = G[j,i] = integrate.quad(lambda x : L(x,i)*L(x,j),lower,upper)[0]
        Flower = np.array([f(lower) for f in F])[:,None]
        return(self.lengthscales/2./self.variance * G + 1./self.variance * np.dot(Flower,Flower.T))

    def dK_dtheta(self,X,X2,target):
        """derivative of the cross-covariance matrix with respect to the parameters (shape is NxMxNparam)"""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:,None,:]-X2[None,:,:])/self.lengthscales),-1))
        invdist = 1./np.where(dist!=0.,dist,np.inf)
        dist2M = np.square(X[:,None,:]-X2[None,:,:])/self.lengthscales**3
        dvar = np.exp(-dist)
        dl = self.variance*dvar[:,:,None]*dist2M*invdist[:,:,np.newaxis]
        np.add(target[:,:,0],dvar, target[:,:,0])
        np.add(target[:,:,1:],dl, target[:,:,1:])
    def dKdiag_dtheta(self,X,target):
        """derivative of the diagonal of the covariance matrix with respect to the parameters (shape is NxNparam)"""
        np.add(target[:,0],1.,target[:,0])
    def dK_dX(self,X,X2,target):
        """derivative of the covariance matrix with respect to X (*! shape is NxMxD !*)."""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:,None,:]-X2[None,:,:])/self.lengthscales),-1))[:,:,None]
        ddist_dX = (X[:,None,:]-X2[None,:,:])/self.lengthscales**2/np.where(dist!=0.,dist,np.inf)
        target += - np.transpose(self.variance*np.exp(-dist)*ddist_dX,(1,0,2))
    def dKdiag_dX(self,X,target):
        pass




        

