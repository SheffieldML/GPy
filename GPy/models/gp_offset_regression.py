# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# Written by Mike Smith. michaeltsmith.org.uk

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern
from ..core import Param

class GPOffsetRegression(GP):
    """
    Gaussian Process model for offset regression

    :param X: input observations, we assume for this class that this has one dimension of actual inputs and the last dimension should be the index of the cluster (so X should be Nx2)
    :param Y: observed values (Nx1?)
    :param kernel: a GPy kernel, defaults to rbf
    :param Norm normalizer: [False]
    :param noise_var: the noise variance for Gaussian likelhood, defaults to 1.

        Normalize Y with the norm given.
        If normalizer is False, no normalization will be done
        If it is None, we use GaussianNorm(alization)

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):

        assert X.shape[1]>1, "Need at least two input dimensions, as last dimension is the label of the cluster"
        if kernel is None:
            kernel = kern.RBF(X.shape[1]-1)

        #self._log_marginal_likelihood = np.nan #todo
        
        likelihood = likelihoods.Gaussian(variance=noise_var)
        self.X_fixed = X[:,:-1]
        self.selected = np.array([int(x) for x in X[:,-1]])
        
        
        super(GPOffsetRegression, self).__init__(X, Y, kernel, likelihood, name='GP offset regression', Y_metadata=Y_metadata, normalizer=normalizer, mean_function=mean_function)
        maxcluster = np.max(self.selected)
        self.offset = Param('offset', np.zeros(maxcluster))
        #self.offset.set_prior(...)
        self.link_parameter(self.offset)
        
    #def dr_doffset(self, X, sel): #how much r changes wrt the offset hyperparameters
        
    #def dL_doffset(self, X, sel):
    #    dL_dr = self.dK_dr_via_X(X, X) * dL_dK
                
        
    def dr_doffset(self,X,sel,delta):
        #given an input matrix, X and the offsets (delta)
        #finds dr/dDelta
        #returns them as a list, one for each offset (delta).        
        #get the input values

        #a matrix G represents the effect of increasing the offset on the radius passed to the kernel for each input. For example
        #what effect will increasing offset 4 have on the kernel output of inputs 5 and 8? Answer: Gs[4][5,8]... (positive or negative)
        Gs = []
        for i,d in enumerate(delta):
            #X[sel==(i+1)]-=d
            G = np.repeat(np.array(sel==(i+1))[:,None]*1,len(X),axis=1) - np.repeat(np.array(sel==(i+1))[None,:]*1,len(X),axis=0)
            Gs.append(G)
        #does subtracting the two Xs end up positive or negative (if negative we need to flip the sign in G).
        w = np.repeat(X,len(X),axis=1) - np.repeat(X.T,len(X),axis=0)
        dr_doffsets = []
        for i,d in enumerate(delta):
            dr_doffset = np.sign(w * Gs[i])
            #print "dr_doffset %d" % i
            #print dr_doffset
            #print Gs[i]
            #print w
            dr_doffsets.append(dr_doffset)
            
        #lastly we need to divide by the lengthscale: So far we've found d(X_i - X_j)/dOffsets
        #we want dr/dOffsets. (X_i - X_j)/lengthscale = r
        dr_doffsets /= self.kern.lengthscale 
        return dr_doffsets
        
    def parameters_changed(self):
        offsets = np.hstack([0.0,self.offset.values])[:,None]
        
        self.X = self.X_fixed - offsets[self.selected]
        super(GPOffsetRegression, self).parameters_changed()
        
        dL_dr = self.kern.dK_dr_via_X(self.X, self.X) * self.grad_dict['dL_dK']
        
        dr_doff = self.dr_doffset(self.X,self.selected,self.offset.values)
        for i in range(len(dr_doff)):
            dL_doff = dL_dr * dr_doff[i]
            self.offset.gradient[i] = -np.sum(dL_doff)
        
