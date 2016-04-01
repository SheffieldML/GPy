# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy, Arno Solin
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Classes in this module enhance Brownian motion covariance function with the
Stochastic Differential Equation (SDE) functionality.
"""

from .brownian import Brownian

import numpy as np

class sde_Brownian(Brownian):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE form.
    
    Linear kernel:

    .. math::

       k(x,y) = \sigma^2 min(x,y)

    """
    
    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """
    
        self.variance.gradient = gradients[0]
        
    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        variance = float(self.variance.values) # this is initial variancve in Bayesian linear regression
        
        F = np.array( ((0,1.0),(0,0) ))
        L = np.array( ((1.0,),(0,)) )
        Qc = np.array( ((variance,),) )
        H = np.array( ((1.0,0),) )
        
        Pinf   = np.array( ( (0, -0.5*variance ), (-0.5*variance, 0) ) )
        #P0 = Pinf.copy() 
        P0 = np.zeros((2,2))   
        #Pinf   = np.array( ( (t0, 1.0), (1.0, 1.0/t0) ) ) * variance
        dF = np.zeros((2,2,1))
        dQc    = np.ones( (1,1,1) )
        
        dPinf = np.zeros((2,2,1))
        dPinf[:,:,0] = np.array( ( (0, -0.5), (-0.5, 0) ) )
        #dP0 = dPinf.copy() 
        dP0 = np.zeros((2,2,1))
  
        return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)
