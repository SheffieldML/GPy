# -*- coding: utf-8 -*-
"""
Classes in this module enhance Matern covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .stationary import Matern32
from .stationary import Matern52
import numpy as np

class sde_Matern32(Matern32):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Matern 3/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{3} r) \exp(- \sqrt{3} r) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        variance = float(self.variance.values)
        lengthscale = float(self.lengthscale.values)
        foo  = np.sqrt(3.)/lengthscale 
        F    = np.array([[0, 1], [-foo**2, -2*foo]]) 
        L    = np.array([[0], [1]]) 
        Qc   = np.array([[12.*np.sqrt(3) / lengthscale**3 * variance]]) 
        H    = np.array([[1, 0]]) 
        Pinf = np.array([[variance, 0],  
        [0,              3.*variance/(lengthscale**2)]]) 
        # Allocate space for the derivatives 
        dF    = np.empty([F.shape[0],F.shape[1],2])
        dQc   = np.empty([Qc.shape[0],Qc.shape[1],2]) 
        dPinf = np.empty([Pinf.shape[0],Pinf.shape[1],2]) 
        # The partial derivatives 
        dFvariance       = np.zeros([2,2]) 
        dFlengthscale    = np.array([[0,0], 
        [6./lengthscale**3,2*np.sqrt(3)/lengthscale**2]]) 
        dQcvariance      = np.array([12.*np.sqrt(3)/lengthscale**3]) 
        dQclengthscale   = np.array([-3*12*np.sqrt(3)/lengthscale**4*variance]) 
        dPinfvariance    = np.array([[1,0],[0,3./lengthscale**2]]) 
        dPinflengthscale = np.array([[0,0], 
        [0,-6*variance/lengthscale**3]]) 
        # Combine the derivatives 
        dF[:,:,0]    = dFvariance 
        dF[:,:,1]    = dFlengthscale 
        dQc[:,:,0]   = dQcvariance 
        dQc[:,:,1]   = dQclengthscale 
        dPinf[:,:,0] = dPinfvariance 
        dPinf[:,:,1] = dPinflengthscale 

        return (F, L, Qc, H, Pinf, dF, dQc, dPinf)

class sde_Matern52(Matern52):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Matern 5/2 kernel:

    .. math::

       k(r) = \sigma^2 (1 + \sqrt{5} r + \frac{5}{3}r^2) \exp(- \sqrt{5} r) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        # Arno, insert your code here

        # Params to use:
        # self.lengthscale
        # self.variance

        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)  