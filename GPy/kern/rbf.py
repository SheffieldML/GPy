# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
import hashlib

class rbf(kernpart):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp(- \frac{1}{2}r^2) \ \ \ \ \  \\text{ where  } r^2 = \sum_{i=1}^d \frac{ (x_i-x^\prime_i)^2}{\ell_i^2}}

    where \ell_i is the lengthscale, \sigma^2 the variance and d the dimensionality of the input.

    :param D: the number of input dimensions
    :type D: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the vector of lengthscale of the kernel
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one single lengthscale parameter \ell), otherwise there is one lengthscale parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    .. Note: this object implements both the ARD and 'spherical' version of the function
    """

    def __init__(self,D,variance=1.,lengthscale=None,ARD=False):
        self.D = D
        self.name = 'rbf'
        self.ARD = ARD
        if not ARD:
            self.Nparam = 2
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                lengthscale = np.ones(1)
        else:
            self.Nparam = self.D + 1
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == self.D, "bad number of lengthscales"
            else:
                lengthscale = np.ones(self.D)

        self._set_params(np.hstack((variance,lengthscale.flatten())))

        #initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3,1))
        self._X, self._X2, self._params = np.empty(shape=(3,1))

    def _get_params(self):
        return np.hstack((self.variance,self.lengthscale))

    def _set_params(self,x):
        assert x.size==(self.Nparam)
        self.variance = x[0]
        self.lengthscale = x[1:]
        self.lengthscale2 = np.square(self.lengthscale)
        #reset cached results
        self._X, self._X2, self._params = np.empty(shape=(3,1))
        self._Z, self._mu, self._S = np.empty(shape=(3,1)) # cached versions of Z,mu,S

    def _get_param_names(self):
        if self.Nparam == 2:
            return ['variance','lengthscale']
        else:
            return ['variance']+['lengthscale_%i'%i for i in range(self.lengthscale.size)]

    def K(self,X,X2,target):
        if X2 is None:
            X2 = X

        self._K_computations(X,X2)
        np.add(self.variance*self._K_dvar, target,target)

    def Kdiag(self,X,target):
        np.add(target,self.variance,target)

    def dK_dtheta(self,partial,X,X2,target):
        self._K_computations(X,X2)
        target[0] += np.sum(self._K_dvar*partial)
        if self.ARD == True:
            dl = self._K_dvar[:,:,None]*self.variance*self._K_dist2/self.lengthscale
            target[1:] += (dl*partial[:,:,None]).sum(0).sum(0)
        else:
            target[1] += np.sum(self._K_dvar*self.variance*(self._K_dist2.sum(-1))/self.lengthscale*partial)
        #np.sum(self._K_dvar*self.variance*self._K_dist2/self.lengthscale*partial)

    def dKdiag_dtheta(self,partial,X,target):
        #NB: derivative of diagonal elements wrt lengthscale is 0
        target[0] += np.sum(partial)

    def dK_dX(self,partial,X,X2,target):
        self._K_computations(X,X2)
        _K_dist = X[:,None,:]-X2[None,:,:]
        dK_dX = np.transpose(-self.variance*self._K_dvar[:,:,np.newaxis]*_K_dist/self.lengthscale2,(1,0,2))
        target += np.sum(dK_dX*partial.T[:,:,None],0)

    def dKdiag_dX(self,partial,X,target):
        pass


    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self,Z,mu,S,target):
        target += self.variance

    def dpsi0_dtheta(self,partial,Z,mu,S,target):
        target[0] += np.sum(partial)

    def dpsi0_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        pass

    def psi1(self,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        target += self._psi1

    def dpsi1_dtheta(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        denom_deriv = S[:,None,:]/(self.lengthscale**3+self.lengthscale*S[:,None,:])
        d_length = self._psi1[:,:,None]*(self.lengthscale*np.square(self._psi1_dist/(self.lengthscale2+S[:,None,:])) + denom_deriv)
        target[0] += np.sum(partial*self._psi1/self.variance)
        dpsi1_dlength = d_length*partial[:,:,None]
        if not self.ARD:
            target[1] += dpsi1_dlength.sum()
        else:
            target[1:] += dpsi1_dlength.sum(0).sum(0)

    def dpsi1_dZ(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        denominator = (self.lengthscale2*(self._psi1_denom))
        dpsi1_dZ = - self._psi1[:,:,None] * ((self._psi1_dist/denominator))
        target += np.sum(partial.T[:,:,None] * dpsi1_dZ, 0)

    def dpsi1_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        self._psi_computations(Z,mu,S)
        tmp = self._psi1[:,:,None]/self.lengthscale2/self._psi1_denom
        target_mu += np.sum(partial.T[:, :, None]*tmp*self._psi1_dist,1)
        target_S += np.sum(partial.T[:, :, None]*0.5*tmp*(self._psi1_dist_sq-1),1)

    def psi2(self,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        target += self._psi2

    def dpsi2_dtheta(self,partial,Z,mu,S,target):
        """Shape N,M,M,Ntheta"""
        self._psi_computations(Z,mu,S)
        d_var = 2.*self._psi2/self.variance
        d_length = self._psi2[:,:,:,None]*(0.5*self._psi2_Zdist_sq*self._psi2_denom + 2.*self._psi2_mudist_sq + 2.*S[:,None,None,:]/self.lengthscale2)/(self.lengthscale*self._psi2_denom)

        target[0] += np.sum(partial*d_var)
        dpsi2_dlength = d_length*partial[:,:,:,None]
        if not self.ARD:
            target[1] += dpsi2_dlength.sum()
        else:
            target[1:] += dpsi2_dlength.sum(0).sum(0).sum(0)
            
    def dpsi2_dZ(self,partial,Z,mu,S,target):
        self._psi_computations(Z,mu,S)
        term1 = 0.5*self._psi2_Zdist/self.lengthscale2 # M, M, Q
        term2 = self._psi2_mudist/self._psi2_denom/self.lengthscale2 # N, M, M, Q
        dZ = self._psi2[:,:,:,None] * (term1[None] + term2)
        target += (partial[:,:,:,None]*dZ).sum(0).sum(0)

    def dpsi2_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        """Think N,M,M,Q """
        self._psi_computations(Z,mu,S)
        tmp = self._psi2[:,:,:,None]/self.lengthscale2/self._psi2_denom
        target_mu += (partial[:,:,:,None]*-tmp*2.*self._psi2_mudist).sum(1).sum(1)
        target_S += (partial[:,:,:,None]*tmp*(2.*self._psi2_mudist_sq-1)).sum(1).sum(1)


    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self,X,X2):
        if not (np.all(X==self._X) and np.all(X2==self._X2)):
            self._X = X
            self._X2 = X2
            if X2 is None: X2 = X
            self._K_dist = X[:,None,:]-X2[None,:,:] # this can be computationally heavy
            self._params = np.empty(shape=(1,0))    #ensure the next section gets called
        if not np.all(self._params == self._get_params()):
            self._params == self._get_params()
            self._K_dist2 = np.square(self._K_dist/self.lengthscale)
            self._K_dvar = np.exp(-0.5*self._K_dist2.sum(-1))

    def _psi_computations(self,Z,mu,S):
        #here are the "statistics" for psi1 and psi2
        if not np.all(Z==self._Z):
            #Z has changed, compute Z specific stuff
            self._psi2_Zhat = 0.5*(Z[:,None,:] +Z[None,:,:]) # M,M,Q
            self._psi2_Zdist = Z[:,None,:]-Z[None,:,:] # M,M,Q
            self._psi2_Zdist_sq = np.square(self._psi2_Zdist)/self.lengthscale2 # M,M,Q
            self._Z = Z

        if not (np.all(Z==self._Z) and np.all(mu==self._mu) and np.all(S==self._S)):
            #something's changed. recompute EVERYTHING

            #TODO: make more efficient for large Q (using NDL's dot product trick)
            #psi1
            self._psi1_denom = S[:,None,:]/self.lengthscale2 + 1.
            self._psi1_dist = Z[None,:,:]-mu[:,None,:]
            self._psi1_dist_sq = np.square(self._psi1_dist)/self.lengthscale2/self._psi1_denom
            self._psi1_exponent = -0.5*np.sum(self._psi1_dist_sq+np.log(self._psi1_denom),-1)
            self._psi1 = self.variance*np.exp(self._psi1_exponent)

            #psi2
            self._psi2_denom = 2.*S[:,None,None,:]/self.lengthscale2+1. # N,M,M,Q
            self._psi2_mudist = mu[:,None,None,:]-self._psi2_Zhat #N,M,M,Q
            self._psi2_mudist_sq = np.square(self._psi2_mudist)/(self.lengthscale2*self._psi2_denom)
            self._psi2_exponent = np.sum(-self._psi2_Zdist_sq/4. -self._psi2_mudist_sq -0.5*np.log(self._psi2_denom),-1) #N,M,M
            self._psi2 = np.square(self.variance)*np.exp(self._psi2_exponent) # N,M,M

            self._Z, self._mu, self._S = Z, mu,S
