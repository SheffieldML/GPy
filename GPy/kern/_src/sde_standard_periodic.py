# -*- coding: utf-8 -*-
"""
Classes in this module enhance Matern covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .standard_periodic import StdPeriodic

import numpy as np

class sde_StdPeriodic(StdPeriodic):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Standard Periodic kernel:

    .. math::

       k(x,y) = \theta_1 \exp \left[  - \frac{1}{2} {}\sum_{i=1}^{input\_dim}  
       \left( \frac{\sin(\frac{\pi}{\lambda_i} (x_i - y_i) )}{l_i} \right)^2 \right] }

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        # Arno, insert your code here

        # Params to use:
        #self.variance
        #self.wavelengths
        #self.lengthscales
        
        # Arno, you could visualize the Latex version of the kernel formula
        # and assume inputs are 1D, so no ARD is used. Then use parameters aboove.        
        
        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)