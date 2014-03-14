# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
import numpy as np
from ...util.linalg import tdot
from ...util.config import *
from stationary import Stationary
from psi_comp import ssrbf_psi_comp

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

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=True, active_dims=None, name='SSRBF'):
        assert ARD==True, "Not Implemented!"
        super(SSRBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
        
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
    
    def psi0(self, Z, variational_posterior):
        ret = np.empty(variational_posterior.mean.shape[0])
        ret[:] = self.variance
        return ret

    def psi1(self, Z, variational_posterior):
        _psi1, _, _, _, _, _, _ = ssrbf_psi_comp._psi1computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
        return _psi1

    def psi2(self, Z, variational_posterior):
        _psi2, _, _, _, _, _, _ = ssrbf_psi_comp._psi2computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
        return _psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        _, _dpsi1_dvariance, _, _, _, _, _dpsi1_dlengthscale = ssrbf_psi_comp._psi1computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
        _, _dpsi2_dvariance, _, _, _, _, _dpsi2_dlengthscale = ssrbf_psi_comp._psi2computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)

        #contributions from psi0:
        self.variance.gradient = np.sum(dL_dpsi0)

        #from psi1
        self.variance.gradient += np.sum(dL_dpsi1 * _dpsi1_dvariance)
        self.lengthscale.gradient = (dL_dpsi1[:,:,None]*_dpsi1_dlengthscale).reshape(-1,self.input_dim).sum(axis=0) 
    

        #from psi2
        self.variance.gradient += (dL_dpsi2 * _dpsi2_dvariance).sum()
        self.lengthscale.gradient += (dL_dpsi2[:,:,:,None] * _dpsi2_dlengthscale).reshape(-1,self.input_dim).sum(axis=0)        
        
    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        _, _, _, _, _, _dpsi1_dZ, _ = ssrbf_psi_comp._psi1computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
        _, _, _, _, _, _dpsi2_dZ, _ = ssrbf_psi_comp._psi2computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)

        #psi1
        grad = (dL_dpsi1[:, :, None] * _dpsi1_dZ).sum(axis=0)

        #psi2
        grad += (dL_dpsi2[:, :, :, None] * _dpsi2_dZ).sum(axis=0).sum(axis=1)

        return grad

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        ndata = variational_posterior.mean.shape[0]
        
        _, _, _dpsi1_dgamma, _dpsi1_dmu, _dpsi1_dS, _, _ = ssrbf_psi_comp._psi1computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)
        _, _, _dpsi2_dgamma, _dpsi2_dmu, _dpsi2_dS, _, _ = ssrbf_psi_comp._psi2computations(self.variance, self.lengthscale, Z, variational_posterior.mean, variational_posterior.variance, variational_posterior.binary_prob)

        #psi1
        grad_mu = (dL_dpsi1[:, :, None] * _dpsi1_dmu).sum(axis=1)
        grad_S = (dL_dpsi1[:, :, None] * _dpsi1_dS).sum(axis=1)
        grad_gamma = (dL_dpsi1[:,:,None] * _dpsi1_dgamma).sum(axis=1)
        #psi2
        grad_mu += (dL_dpsi2[:, :, :, None] * _dpsi2_dmu).reshape(ndata,-1,self.input_dim).sum(axis=1)
        grad_S += (dL_dpsi2[:, :, :, None] * _dpsi2_dS).reshape(ndata,-1,self.input_dim).sum(axis=1)
        grad_gamma += (dL_dpsi2[:,:,:, None] * _dpsi2_dgamma).reshape(ndata,-1,self.input_dim).sum(axis=1)
        
        return grad_mu, grad_S, grad_gamma
        
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

