# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# Written by Mike Smith. michaeltsmith.org.uk

from __future__ import division
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import math

class Offset(Kern):
    """
    A kernel in which subsets of the data can be shifted.
    
    :offsets list of offsets of length of number of subsets
    """
    def __init__(self, kernel, input_dim, offsets=[], index_dim=-1, active_dims=None, name='offset'):
        super(Offset, self).__init__(input_dim, active_dims, name)
        assert isinstance(index_dim, int), "Offset kernel is only defined with one input dimension being the index"
        self.kern = kernel
        
        self.offsets = Param('offset', offsets)
        self.link_parameter(self.offsets)
        
        self.offset_index_dim = index_dim
        self.link_parameters(kernel) #maybe whole class should inherit from CombinationKernel?

    def shift(self,X):
        self.selected = np.array([int(x) for x in X[:,self.offset_index_dim]])
        offsets = np.hstack([0.0,self.offsets.values])[:,None]
        return X - offsets[self.selected]
        
    def K(self,X ,X2=None):
        return self.kern.K(self.shift(X),X2)

    def Kdiag(self,X):
        return self.kern.Kdiag(self.shift(X))
        
        
        
    def dX_doffset(self,sel,delta):
        #given the select array (sel) and the offsets (delta)
        #finds dX/dDelta
        #returns them as a 2d matrix, one row/col[?] for each offset (delta).

        #a matrix G represents the effect of increasing the offset on the X_i passed to the kernel for each input. For example
        #what effect will increasing offset 4 have on the kernel output of input 5? Answer: Gs[4,5]... (positive or zero)
        Gs = []
        for i,d in enumerate(delta):            
            G = np.array(sel==(i+1))[:,None]*1
            Gs.append(G)      
        return np.array(Gs)*2.0
            
    def update_gradients_full(self,dL_dK,X,X2=None):
        dL_dX = self.kern.gradients_X(dL_dK, self.shift(X), X)
        dX_doff = self.dX_doffset(self.selected,self.offsets.values)
        for i in range(len(dX_doff)):
            dL_doff = dL_dX * dX_doff[i]
            self.offsets.gradient[i] = -np.sum(dL_doff)
        self.kern.update_gradients_full(dL_dK,self.shift(X),X2)
        
    def gradients_X(self,dL_dK, X, X2=None):
        return self.kern.gradients_X(dL_dK,self.shift(X),X2)

    def gradients_X_diag(self, dL_dKdiag, X):
        return self.kern.gradients_X_diag(dL_dKdiag, self.shift(X))
        
    def update_gradients_diag(self, dL_dKdiag, X): #TODO What is this?
        raise NotImplementedError("Not implemented Offset.update_gradients_diag")
        #self.kern.update_gradients_diag(dL_dKdiag, self.shift(X))

