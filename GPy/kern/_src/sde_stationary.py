# -*- coding: utf-8 -*-
"""
Classes in this module enhance several stationary covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .rbf import RBF
from .stationary import Exponential
from .stationary import RatQuad

import numpy as np

class sde_RBF(RBF):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Radial Basis Function kernel:

    .. math::

        k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

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

class sde_Exponential(Exponential):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Exponential kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r \\bigg) \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        F  = np.array([[-1/self.lengthscale]]) 
        L  = np.array([[1]]) 
        Qc = np.array([[2*self.variance/self.lengthscale]]) 
        H = np.array([[1]]) 
        Pinf = np.array([[self.variance]]) 
        # TODO: return the derivatives as well 
        
        return (F, L, Qc, H, Pinf)
        
        # Arno, insert your code here
        
        # Params to use:

        # self.lengthscale
        # self.variance

        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf) 

class sde_RatQuad(RatQuad):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Rational Quadratic kernel:

    .. math::

       k(r) = \sigma^2 \\bigg( 1 + \\frac{r^2}{2} \\bigg)^{- \alpha} \\ \\ \\ \\  \text{ where  } r = \sqrt{\sum_{i=1}^{input dim} \frac{(x_i-y_i)^2}{\ell_i^2} }

    """

    def sde(self):
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        # Arno, insert your code here
        
        # Params to use:

        # self.lengthscale
        # self.variance
        #self.power
        
        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)  