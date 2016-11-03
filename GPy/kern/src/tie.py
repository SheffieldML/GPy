# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# Written by Mike Smith. michaeltsmith.org.uk

from __future__ import division
import numpy as np
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
import math

class Tie(Kern):
    """
    A kernel which takes another kernel and ties its parameters together
    
    :tied_param_list a list of lists of parameters to tie together. Each item
                    in the inner list is a regex, so one could use
                     ['independ.offset.Mat32.lengthscale',
                      'independ.Mat32_1.lengthscale',
                      'independ.Mat32.lengthscale']
                    or just
                     ['.*lengthscale']
    """
    def __init__(self, kernel, input_dim, tied_param_list, active_dims=None, name='tie'):
           #TODO We need to initialise the values of the parameters to be equal!
        super(Tie, self).__init__(input_dim, active_dims, name)
        self.kern = kernel

        self.params = [] #list of parameter objects to tie together

        for tlist in tied_param_list:
            plist = []  #temp array for list of parameter objects for each tie
            assert type(tlist) is list, "The tied_param_list should be a list of lists of strings"
            for t in tlist: #expand regex in each inner list and add all matches
                plist.extend(self.kern.grep_param_names(t)) 
                
            if len(plist)==0:
                print("Warning: No parameters were added for (%s)" % str(tlist))
            else:
                self.params.append(plist)
            
        self.link_parameters(self.kern)  
        
        for pitem in self.params:
            l = len(pitem)
            v,g = self.get_totals(pitem)
            for p in pitem:
                p.param_array[:] = v/l   

            
    def get_totals(self,param_list):
        """
        Returns the sum total of the gradients and values of the parameters in
        the param_list
        """
        v = None
        g = None
        for p in param_list:
            if v is None:
                v = p.values.copy()
                g = p.gradient.copy()
            else:
                v += p.values
                g += p.gradient
        return v,g
    
    def update_gradients_full(self,dL_dK,X,X2=None):
        self.kern.update_gradients_full(dL_dK,X,X2)
        for pitem in self.params:
            l = len(pitem)
            v,g = self.get_totals(pitem)
            for p in pitem:
                #p.param_array[:] = v/l #TODO: Just do once in __init__
                p.gradient = g/l #pitem['main'].gradient
            
    def gradients_X(self,dL_dK, X, X2=None):
        return self.kern.gradients_X(dL_dK, X, X2=None)
        
    def gradients_X_diag(self, dL_dKdiag, X):
        return self.kern.gradients_X_diag(dL_dKdiag, X)

    def K(self,X ,X2=None):
        return self.kern.K(X,X2)

    def Kdiag(self,X):
        return self.kern.Kdiag(X)

    def get_ties_names(self,html=False):
        textlist = []
        if html:
            lb = "<br />"
        else:
            lb = "\n"
        for ps in self.params:
            innerlist = []
            for p in ps:
                innerlist.append(p.hierarchy_name())
            textlist.append(innerlist)
        st = lb+lb
        st += "The following sets of parameters are tied:"
        for texts in textlist:
            st += lb
            st += lb.join(texts)
            st += lb
        return st
                
    def __str__(self, header=True, VT100=True):
        st = super(Tie, self).__str__(header, VT100)
        st += self.get_ties_names()
        return st
        
    def _repr_html_(self, header=True):
        toprint = super(Tie,self)._repr_html_(header)
        toprint+= self.get_ties_names(html=True)
        return toprint
