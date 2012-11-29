# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib
from ..util.linalg import pdinv,mdot
from scipy import integrate

class Matern32(kernpart):
    """
    Matern 3/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{3} r) \exp(- \sqrt{3} r) \qquad \qquad \\text{ where  } r = \sqrt{\sum_{i=1}^D \\frac{(x_i-y_i)^2}{\ell_i^2} }

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
        self.name = 'Mat32'
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
        np.add(self.variance*(1+np.sqrt(3.)*dist)*np.exp(-np.sqrt(3.)*dist), target,target)
    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        np.add(target,self.variance,target)

    def Gram_matrix(self,F,F1,F2,lower,upper):
        """
        Return the Gram matrix of the vector of functions F with respect to the RKHS norm. The use of this function is limited to D=1.

        :param F: vector of functions
        :type F: np.array
        :param F1: vector of derivatives of F
        :type F1: np.array
        :param F2: vector of second derivatives of F
        :type F2: np.array    
        :param lower,upper: boundaries of the input domain
        :type lower,upper: floats  
        """
        assert self.D == 1
        def L(x,i):
            return(3./self.lengthscales**2*F[i](x) + 2*np.sqrt(3)/self.lengthscales*F1[i](x) + F2[i](x))
        n = F.shape[0]
        G = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                G[i,j] = G[j,i] = integrate.quad(lambda x : L(x,i)*L(x,j),lower,upper)[0]
        Flower = np.array([f(lower) for f in F])[:,None]
        F1lower = np.array([f(lower) for f in F1])[:,None]
        #print "OLD \n", np.dot(F1lower,F1lower.T), "\n \n"
        #return(G)
        return(self.lengthscales**3/(12.*np.sqrt(3)*self.variance) * G + 1./self.variance*np.dot(Flower,Flower.T) + self.lengthscales**2/(3.*self.variance)*np.dot(F1lower,F1lower.T))

    def dK_dtheta(self,X,X2,target):
        """derivative of the cross-covariance matrix with respect to the parameters (shape is NxMxNparam)"""
        if X2 is None: X2 = X
        dist = np.sqrt(np.sum(np.square((X[:,None,:]-X2[None,:,:])/self.lengthscales),-1))
        dvar = (1+np.sqrt(3.)*dist)*np.exp(-np.sqrt(3.)*dist)
        invdist = 1./np.where(dist!=0.,dist,np.inf)
        dist2M = np.square(X[:,None,:]-X2[None,:,:])/self.lengthscales**3
        dl = (self.variance* 3 * dist * np.exp(-np.sqrt(3.)*dist))[:,:,np.newaxis] * dist2M*invdist[:,:,np.newaxis]
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
        target += -  np.transpose(3*self.variance*dist*np.exp(-np.sqrt(3)*dist)*ddist_dX,(1,0,2))
    def dKdiag_dX(self,X,target):
        pass

