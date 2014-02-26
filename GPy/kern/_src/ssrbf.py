# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
import numpy as np
from ...util.linalg import tdot
from ...util.config import *
from stationary import Stationary

class SSRBF(Stationary):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel
    for Spike-and-Slab GPLVM

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg) \ \ \ \ \  \\text{ where  } r^2 = \sum_{i=1}^d \\frac{ (x_i-x^\prime_i)^2}{\ell_i^2}

    where \ell_i is the lengthscale, \sigma^2 the variance and d the dimensionality of the input.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the vector of lengthscale of the kernel
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one single lengthscale parameter \ell), otherwise there is one lengthscale parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    .. Note: this object implements both the ARD and 'spherical' version of the function
    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=True, name='SSRBF'):
        assert ARD==True, "Not Implemented!"
        super(SSRBF, self).__init__(input_dim, variance, lengthscale, ARD, name)
        
    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r**2)

    def dK_dr(self, r):
        return -r*self.K_of_r(r)

    def parameters_changed(self):
        pass

    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret
        
    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#
    
    def psi0(self, Z, posterior_variational):
        ret = np.empty(posterior_variational.mean.shape[0])
        ret[:] = self.variance
        return ret

    def psi1(self, Z, posterior_variational):
        self._psi_computations(Z, posterior_variational.mean, posterior_variational.variance, posterior_variational.binary_prob)
        return self._psi1

    def psi2(self, Z, posterior_variational):
        self._psi_computations(Z, posterior_variational.mean, posterior_variational.variance, posterior_variational.binary_prob)
        return self._psi2

    def dL_dpsi0_dmuSgamma(self, dL_dpsi0, Z, mu, S, gamma, target_mu, target_S, target_gamma):
        pass


    def dL_dpsi1_dmuSgamma(self, dL_dpsi1, Z, mu, S, gamma, target_mu, target_S, target_gamma):
        self._psi_computations(Z, mu, S, gamma)
        target_mu += (dL_dpsi1[:, :, None] * self._dpsi1_dmu).sum(axis=1)
        target_S += (dL_dpsi1[:, :, None] * self._dpsi1_dS).sum(axis=1)
        target_gamma += (dL_dpsi1[:,:,None] * self._dpsi1_dgamma).sum(axis=1)


    def dL_dpsi2_dmuSgamma(self, dL_dpsi2, Z, mu, S, gamma, target_mu, target_S, target_gamma):
        """Think N,num_inducing,num_inducing,input_dim """
        self._psi_computations(Z, mu, S, gamma)
        target_mu += (dL_dpsi2[:, :, :, None] * self._dpsi2_dmu).reshape(mu.shape[0],-1,mu.shape[1]).sum(axis=1)
        target_S += (dL_dpsi2[:, :, :, None] * self._dpsi2_dS).reshape(S.shape[0],-1,S.shape[1]).sum(axis=1)
        target_gamma += (dL_dpsi2[:,:,:, None] *self._dpsi2_dgamma).reshape(gamma.shape[0],-1,gamma.shape[1]).sum(axis=1)

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, posterior_variational):
        self._psi_computations(Z, posterior_variational.mean, posterior_variational.variance, posterior_variational.binary_prob)

        #contributions from psi0:
        self.variance.gradient = np.sum(dL_dpsi0)

        #from psi1
        self.variance.gradient += np.sum(dL_dpsi1 * self._dpsi1_dvariance)
        self.lengthscale.gradient = (dL_dpsi1[:,:,None]*self._dpsi1_dlengthscale).reshape(-1,self.input_dim).sum(axis=0) 
    

        #from psi2
        self.variance.gradient += (dL_dpsi2 * self._dpsi2_dvariance).sum()
        self.lengthscale.gradient += (dL_dpsi2[:,:,:,None] * self._dpsi2_dlengthscale).reshape(-1,self.input_dim).sum(axis=0)

        #from Kmm
        self._K_computations(Z, None)
        dvardLdK = self._K_dvar * dL_dKmm
        var_len3 = self.variance / (np.square(self.lengthscale)*self.lengthscale)

        self.variance.gradient += np.sum(dvardLdK)
        self.lengthscale.gradient += (np.square(Z[:,None,:]-Z[None,:,:])*dvardLdK[:,:,None]).reshape(-1,self.input_dim).sum(axis=0)*var_len3
        
        
    def gradients_Z_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, posterior_variational):
        self._psi_computations(Z, posterior_variational.mean, posterior_variational.variance, posterior_variational.binary_prob)

        #psi1
        grad = (dL_dpsi1[:, :, None] * self._dpsi1_dZ).sum(axis=0)

        #psi2
        grad += (dL_dpsi2[:, :, :, None] * self._dpsi2_dZ).sum(axis=0).sum(axis=1)

        grad += self.gradients_X(dL_dKmm, Z, None)

        return grad

    def gradients_q_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, posterior_variational):
        ndata = posterior_variational.mean.shape[0]
        self._psi_computations(Z, posterior_variational.mean, posterior_variational.variance, posterior_variational.binary_prob)
        #psi1
        grad_mu = (dL_dpsi1[:, :, None] * self._dpsi1_dmu).sum(axis=1)
        grad_S = (dL_dpsi1[:, :, None] * self._dpsi1_dS).sum(axis=1)
        grad_gamma = (dL_dpsi1[:,:,None] * self._dpsi1_dgamma).sum(axis=1)
        #psi2
        grad_mu += (dL_dpsi2[:, :, :, None] * self._dpsi2_dmu).reshape(ndata,-1,self.input_dim).sum(axis=1)
        grad_S += (dL_dpsi2[:, :, :, None] * self._dpsi2_dS).reshape(ndata,-1,self.input_dim).sum(axis=1)
        grad_gamma += (dL_dpsi2[:,:,:, None] *self._dpsi2_dgamma).reshape(ndata,-1,self.input_dim).sum(axis=1)
        
        return grad_mu, grad_S, grad_gamma

    def gradients_X(self, dL_dK, X, X2=None):
        #if self._X is None or X.base is not self._X.base or X2 is not None:
        if X2==None:
            _K_dist = X[:,None,:] - X[None,:,:]
            _K_dist2 = np.square(_K_dist/self.lengthscale).sum(axis=-1)
            dK_dX = self.variance*np.exp(-0.5 * self._K_dist2[:,:,None]) * (-2.*_K_dist/np.square(self.lengthscale))
            dL_dX = (dL_dK[:,:,None] * dK_dX).sum(axis=1)
        else:
            _K_dist = X[:,None,:] - X2[None,:,:]
            _K_dist2 = np.square(_K_dist/self.lengthscale).sum(axis=-1)
            dK_dX = self.variance*np.exp(-0.5 * self._K_dist2[:,:,None]) * (-_K_dist/np.square(self.lengthscale))
            dL_dX = (dL_dK[:,:,None] * dK_dX).sum(axis=1)
        return dL_dX
        
    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    #@cache_this(1)
    def _K_computations(self, X, X2):
        """
        K(X,X2) - X is NxQ 
        Q -> input dimension (self.input_dim)
        """
        if X2 is None:
            self._X2 = None
                            
            X = X / self.lengthscale
            Xsquare = np.sum(np.square(X), axis=1)
            self._K_dist2 = -2.*tdot(X) + (Xsquare[:, None] + Xsquare[None, :])
        else:
            self._X2 = X2.copy()
            
            X = X / self.lengthscale
            X2 = X2 / self.lengthscale
            self._K_dist2 = -2.*np.dot(X, X2.T) + (np.sum(np.square(X), axis=1)[:, None] + np.sum(np.square(X2), axis=1)[None, :])
        self._K_dvar = np.exp(-0.5 * self._K_dist2)

    #@cache_this(1)
    def _psi_computations(self, Z, mu, S, gamma):
        """
        Z - MxQ
        mu - NxQ
        S - NxQ
        gamma - NxQ
        """
        # here are the "statistics" for psi1 and psi2
        # Produced intermediate results:
        # _psi1                NxM
        # _dpsi1_dvariance     NxM
        # _dpsi1_dlengthscale  NxMxQ
        # _dpsi1_dZ            NxMxQ
        # _dpsi1_dgamma        NxMxQ
        # _dpsi1_dmu           NxMxQ
        # _dpsi1_dS            NxMxQ
        # _psi2                NxMxM
        # _psi2_dvariance      NxMxM
        # _psi2_dlengthscale   NxMxMxQ
        # _psi2_dZ             NxMxMxQ
        # _psi2_dgamma         NxMxMxQ
        # _psi2_dmu            NxMxMxQ
        # _psi2_dS             NxMxMxQ
        
        lengthscale2 = np.square(self.lengthscale)
                    
        _psi2_Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
        _psi2_Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
        _psi2_Zdist_sq = np.square(_psi2_Zdist / self.lengthscale) # M,M,Q
        _psi2_Z_sq_sum = (np.square(Z[:,None,:])+np.square(Z[None,:,:]))/lengthscale2 # MxMxQ

        # psi1
        _psi1_denom = S[:, None, :] / lengthscale2 + 1.  # Nx1xQ
        _psi1_denom_sqrt = np.sqrt(_psi1_denom) #Nx1xQ
        _psi1_dist = Z[None, :, :] - mu[:, None, :]  # NxMxQ
        _psi1_dist_sq = np.square(_psi1_dist) / (lengthscale2 * _psi1_denom) # NxMxQ
        _psi1_common = gamma[:,None,:] / (lengthscale2*_psi1_denom*_psi1_denom_sqrt) #Nx1xQ
        _psi1_exponent1 = np.log(gamma[:,None,:]) -0.5 * (_psi1_dist_sq + np.log(_psi1_denom)) # NxMxQ
        _psi1_exponent2 = np.log(1.-gamma[:,None,:]) -0.5 * (np.square(Z[None,:,:])/lengthscale2) # NxMxQ
        _psi1_exponent = np.log(np.exp(_psi1_exponent1) + np.exp(_psi1_exponent2)) #NxMxQ
        _psi1_exp_sum = _psi1_exponent.sum(axis=-1) #NxM
        _psi1_exp_dist_sq = np.exp(-0.5*_psi1_dist_sq) # NxMxQ
        _psi1_exp_Z = np.exp(-0.5*np.square(Z[None,:,:])/lengthscale2) # 1xMxQ
        _psi1_q = self.variance * np.exp(_psi1_exp_sum[:,:,None] - _psi1_exponent) # NxMxQ
        self._psi1 = self.variance * np.exp(_psi1_exp_sum) # NxM
        self._dpsi1_dvariance = self._psi1 / self.variance # NxM
        self._dpsi1_dgamma = _psi1_q * (_psi1_exp_dist_sq/_psi1_denom_sqrt-_psi1_exp_Z) # NxMxQ
        self._dpsi1_dmu = _psi1_q * (_psi1_exp_dist_sq * _psi1_dist * _psi1_common) # NxMxQ
        self._dpsi1_dS = _psi1_q * (_psi1_exp_dist_sq * _psi1_common * 0.5 * (_psi1_dist_sq - 1.)) # NxMxQ
        self._dpsi1_dZ = _psi1_q * (- _psi1_common * _psi1_dist * _psi1_exp_dist_sq - (1-gamma[:,None,:])/lengthscale2*Z[None,:,:]*_psi1_exp_Z) # NxMxQ
        self._dpsi1_dlengthscale = 2.*self.lengthscale*_psi1_q * (0.5*_psi1_common*(S[:,None,:]/lengthscale2+_psi1_dist_sq)*_psi1_exp_dist_sq + 0.5*(1-gamma[:,None,:])*np.square(Z[None,:,:]/lengthscale2)*_psi1_exp_Z) # NxMxQ


        # psi2
        _psi2_denom = 2.*S[:, None, None, :] / lengthscale2 + 1. # Nx1x1xQ
        _psi2_denom_sqrt = np.sqrt(_psi2_denom)
        _psi2_mudist = mu[:,None,None,:]-_psi2_Zhat #N,M,M,Q
        _psi2_mudist_sq = np.square(_psi2_mudist)/(lengthscale2*_psi2_denom)
        _psi2_common = gamma[:,None,None,:]/(lengthscale2 * _psi2_denom * _psi2_denom_sqrt) # Nx1x1xQ
        _psi2_exponent1 = -_psi2_Zdist_sq -_psi2_mudist_sq -0.5*np.log(_psi2_denom)+np.log(gamma[:,None,None,:]) #N,M,M,Q
        _psi2_exponent2 = np.log(1.-gamma[:,None,None,:]) - 0.5*(_psi2_Z_sq_sum) # NxMxMxQ
        _psi2_exponent = np.log(np.exp(_psi2_exponent1) + np.exp(_psi2_exponent2))
        _psi2_exp_sum = _psi2_exponent.sum(axis=-1) #NxM
        _psi2_q = np.square(self.variance) * np.exp(_psi2_exp_sum[:,:,:,None]-_psi2_exponent) # NxMxMxQ 
        _psi2_exp_dist_sq = np.exp(-_psi2_Zdist_sq -_psi2_mudist_sq) # NxMxMxQ
        _psi2_exp_Z = np.exp(-0.5*_psi2_Z_sq_sum) # MxMxQ
        self._psi2 = np.square(self.variance) * np.exp(_psi2_exp_sum) # N,M,M
        self._dpsi2_dvariance = 2. * self._psi2/self.variance # NxMxM
        self._dpsi2_dgamma = _psi2_q * (_psi2_exp_dist_sq/_psi2_denom_sqrt - _psi2_exp_Z) # NxMxMxQ
        self._dpsi2_dmu = _psi2_q * (-2.*_psi2_common*_psi2_mudist * _psi2_exp_dist_sq) # NxMxMxQ
        self._dpsi2_dS = _psi2_q * (_psi2_common * (2.*_psi2_mudist_sq - 1.) * _psi2_exp_dist_sq) # NxMxMxQ
        self._dpsi2_dZ = 2.*_psi2_q * (_psi2_common*(-_psi2_Zdist*_psi2_denom+_psi2_mudist)*_psi2_exp_dist_sq - (1-gamma[:,None,None,:])*Z[:,None,:]/lengthscale2*_psi2_exp_Z) # NxMxMxQ
        self._dpsi2_dlengthscale = 2.*self.lengthscale* _psi2_q * (_psi2_common*(S[:,None,None,:]/lengthscale2+_psi2_Zdist_sq*_psi2_denom+_psi2_mudist_sq)*_psi2_exp_dist_sq+(1-gamma[:,None,None,:])*_psi2_Z_sq_sum*0.5/lengthscale2*_psi2_exp_Z) # NxMxMxQ
        