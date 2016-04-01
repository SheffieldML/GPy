# -*- coding: utf-8 -*-
# Copyright (c) 2015, Alex Grigorevskiy, Arno Solin
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Classes in this module enhance Linear covariance function with the
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
    def __init__(self, input_dim, X, variances=None, ARD=False, active_dims=None, name='linear'):
        """
        Modify the init method, because one extra parameter is required. X - points
        on the X axis.
        """
        
        super(sde_Linear, self).__init__(input_dim, variances, ARD, active_dims, name)
        
        self.t0 = np.min(X)
        
    
    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """
    
        self.variances.gradient = gradients[0]
        
    def sde(self): 
        """ 
        Return the state space representation of the covariance. 
        """ 
        
        variance = float(self.variances.values) # this is initial variancve in Bayesian linear regression
        t0 = float(self.t0)
        
        F = np.array( ((0,1.0),(0,0) ))
        L = np.array( ((0,),(1.0,)) )
        Qc = np.zeros((1,1))
        H = np.array( ((1.0,0),) )
        
        Pinf   = np.zeros((2,2))
        P0 = np.array( ( (t0**2, t0), (t0, 1) ) ) * variance        
        dF = np.zeros((2,2,1))
        dQc    = np.zeros( (1,1,1) )
        
        dPinf = np.zeros((2,2,1))
        dP0 = np.zeros((2,2,1))
        dP0[:,:,0]  = P0 / variance
  
        return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)
