# -*- coding: utf-8 -*-

# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
The standard periodic kernel which mentioned in:

[1] Gaussian Processes for Machine Learning, C. E. Rasmussen, C. K. I. Williams.
The MIT Press, 2005.


[2] Introduction to Gaussian processes. D. J. C. MacKay. In C. M. Bishop, editor, 
Neural Networks and Machine Learning, pages 133-165. Springer, 1998.
"""

from .kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp

import numpy as np

class StdPeriodic(Kern):
    """
    Standart periodic kernel

    .. math::

       k(x,y) = \theta_1 \exp \left[  - \frac{1}{2} {}\sum_{i=1}^{input\_dim}  
       \left( \frac{\sin(\frac{\pi}{\lambda_i} (x_i - y_i) )}{l_i} \right)^2 \right] }

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance :math:`\theta_1` in the formula above
    :type variance: float
    :param wavelength: the vector of wavelengths :math:`\lambda_i`. If None then 1.0 is assumed.
    :type wavelength: array or list of the appropriate size (or float if there is only one wavelength parameter)
    :param lengthscale: the vector of lengthscale :math:`\l_i`. If None then 1.0 is assumed.
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD1: Auto Relevance Determination with respect to wavelength. 
        If equal to "False" one single wavelength parameter :math:`\lambda_i` for 
        each dimension is assumed, otherwise there is one lengthscale 
        parameter per dimension.
    :type ARD1: Boolean
    :param ARD2: Auto Relevance Determination with respect to lengthscale. 
        If equal to "False" one single wavelength parameter :math:`l_i` for 
        each dimension is assumed, otherwise there is one lengthscale 
        parameter per dimension.
    :type ARD2: Boolean
    :param active_dims: indices of dimensions which are used in the computation of the kernel
    :type wavelength: array or list of the appropriate size
    :param name: Name of the kernel for output
    :type String
    :param useGPU: whether of not use GPU
    :type Boolean
    """
    
    def __init__(self, input_dim, variance=1., wavelength=None, lengthscale=None, ARD1=False, ARD2=False, active_dims=None, name='std_periodic',useGPU=False):
        super(StdPeriodic, self).__init__(input_dim, active_dims, name, useGPU=useGPU)
        self.input_dim = input_dim
        self.ARD1 = ARD1 # correspond to wavelengths        
        self.ARD2 = ARD2 # correspond to lengthscales
        
        self.name = name
        
        if self.ARD1 == False:
            if wavelength is not None:
                wavelength = np.asarray(wavelength)
                assert wavelength.size == 1, "Only one wavelength needed for non-ARD kernel"
            else:
                wavelength = np.ones(1)
        else:
            if wavelength is not None:
                wavelength = np.asarray(wavelength)
                assert wavelength.size == input_dim, "bad number of wavelengths"
            else:
                wavelength = np.ones(input_dim)
        
        if self.ARD2 == False:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                lengthscale = np.ones(1)
        else:
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == input_dim, "bad number of lengthscales"
            else:
                lengthscale = np.ones(input_dim)
        
        self.variance = Param('variance', variance, Logexp())
        assert self.variance.size==1, "Variance size must be one"
        self.wavelengths =  Param('wavelengths', wavelength, Logexp())
        self.lengthscales =  Param('lengthscales', lengthscale, Logexp())
        
        self.link_parameters(self.variance,  self.wavelengths, self.lengthscales)

    def parameters_changed(self):
        """
        This functions deals as a callback for each optimization iteration. 
        If one optimization step was successfull and the parameters
        this callback function will be called to be able to update any 
        precomputations for the kernel.
        """
        
        pass
        
        
    def K(self, X, X2=None):
        """Compute the covariance matrix between X and X2."""
        if X2 is None: 
            X2 = X
            
        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.wavelengths
        exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / self.lengthscales ), axis = -1 ) ) 
            
        return self.variance * exp_dist


    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix associated to X."""
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def update_gradients_full(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None: 
            X2 = X
        
        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.wavelengths
        
        sin_base = np.sin( base )         
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / self.lengthscales ), axis = -1 ) ) 
        
        dwl = self.variance * (1.0/np.square(self.lengthscales)) * sin_base*np.cos(base) * (base / self.wavelengths)
        
        dl = self.variance * np.square( sin_base) / np.power( self.lengthscales, 3) 
        
        self.variance.gradient = np.sum(exp_dist * dL_dK)    
        #target[0] += np.sum( exp_dist * dL_dK)        
        
        if self.ARD1: # different wavelengths
            self.wavelengths.gradient = (dwl * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else:  # same wavelengths
            self.wavelengths.gradient = np.sum(dwl.sum(-1) * exp_dist * dL_dK)
            
        if self.ARD2: # different lengthscales
            self.lengthscales.gradient = (dl * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else: # same lengthscales
            self.lengthscales.gradient = np.sum(dl.sum(-1) * exp_dist * dL_dK)
        
    def update_gradients_diag(self, dL_dKdiag, X):
        """derivative of the diagonal of the covariance matrix with respect to the parameters."""
        self.variance.gradient = np.sum(dL_dKdiag)
        self.wavelengths.gradient = 0
        self.lengthscales.gradient = 0

    def gradients_X(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None:
            X2 = X

        base = np.pi * (X[:, None, :] - X2[None, :, :]) / self.wavelengths

        sin_base = np.sin( base )
        exp_dist = np.exp( -0.5* np.sum( np.square(  sin_base / self.lengthscales ), axis = -1 ) )

        dx = -self.variance * (np.pi / (self.wavelengths*np.square(self.lengthscales))) * sin_base*np.cos(base)

        if self.ARD1:  # different wavelengths
            return (dx * exp_dist[:,:,None] * dL_dK[:, :, None]).sum(0).sum(0)
        else:  # same wavelengths
            return np.sum(dx.sum(-1) * exp_dist * dL_dK)
#        raise NotImplemented("Periodic kernel: dK_dX not implemented")
#
    def gradients_X_diag(self, dL_dKdiag, X):
        """derivative of the covariance matrix with respect to X - diagonal."""
        pass  # diagonal element is zero
#        raise NotImplemented("Periodic kernel: dKdiag_dX not implemented")
