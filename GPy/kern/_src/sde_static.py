# -*- coding: utf-8 -*-
"""
Classes in this module enhance Matern covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .static import White
from .static import Bias

import numpy as np

class sde_White(White):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Linear kernel:

    .. math::

       k(x,y) = \alpha

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        # Arno, insert your code here

        # Params to use:
        # self.variance

        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)

class sde_Bias(Bias):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Linear kernel:

    .. math::

       k(x,y) = \alpha*\delta(x-y)

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        # Arno, insert your code here
        
        # Params to use:
        # self.variance
        
        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)