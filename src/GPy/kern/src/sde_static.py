# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy, Arno Solin
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Classes in this module enhance Static covariance functions with the
Stochastic Differential Equation (SDE) functionality.
"""
from .static import White
from .static import Bias

import numpy as np

class sde_White(White):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    White kernel:

    .. math::

       k(x,y) = \alpha*\delta(x-y)

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
        
        variance = float(self.variance.values) 
        
        F = np.array( ((-np.inf,),) )
        L = np.array( ((1.0,),)  )
        Qc = np.array( ((variance,),)  )
        H = np.array( ((1.0,),) )
        
        Pinf   = np.array( ((variance,),)  )
        P0 = Pinf.copy()     
        
        dF = np.zeros((1,1,1))
        dQc = np.zeros((1,1,1))
        dQc[:,:,0]    = np.array( ((1.0,),) )
        
        dPinf = np.zeros((1,1,1))
        dPinf[:,:,0] = np.array( ((1.0,),) )
        dP0 = dPinf.copy()
        
        return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)


class sde_Bias(Bias):
    """
    
    Class provide extra functionality to transfer this covariance function into
    SDE forrm.
    
    Bias kernel:

    .. math::

       k(x,y) = \alpha

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
        variance = float(self.variance.values) 
        
        F = np.array( ((0.0,),))
        L = np.array( ((1.0,),))
        Qc = np.zeros((1,1))
        H = np.array( ((1.0,),))
        
        Pinf   = np.zeros((1,1))
        P0 = np.array( ((variance,),) )      
        
        dF = np.zeros((1,1,1))
        dQc    = np.zeros((1,1,1))
        
        dPinf = np.zeros((1,1,1))
        dP0 = np.zeros((1,1,1))
        dP0[:,:,0] = np.array( ((1.0,),) )
        
        return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)