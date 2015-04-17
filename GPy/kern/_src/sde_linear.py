# -*- coding: utf-8 -*-
"""
Classes in this module enhance Matern covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .linear import Linear

import numpy as np

class sde_Linear(Linear):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Linear kernel:

    .. math::

       k(x,y) = \sum_{i=1}^{input dim} \sigma^2_i x_iy_i

    """

    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        # Arno, insert your code here

        # Params to use:

        # self.variances

        #return (F, L, Qc, H, Pinf, dF, dQc, dPinf)
