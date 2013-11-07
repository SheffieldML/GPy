# Copyright (c) 2012, James Hesnsman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
from independent_outputs import index_to_slices

class Hierarchical(Kernpart):
    """
    A kernel part which can reopresent a hierarchy of indepencnce: a generalisation of independent_outputs

    """
    def __init__(self,parts):
        self.levels = len(parts)
        self.input_dim = parts[0].input_dim + 1
        self.num_params = np.sum([k.num_params for k in parts])
        self.name = 'hierarchy'
        self.parts = parts

        self.param_starts = np.hstack((0,np.cumsum([k.num_params for k in self.parts[:-1]])))
        self.param_stops = np.cumsum([k.num_params for k in self.parts])

    def _get_params(self):
        return np.hstack([k._get_params() for k in self.parts])

    def _set_params(self,x):
        [k._set_params(x[start:stop]) for k, start, stop in zip(self.parts, self.param_starts, self.param_stops)]

    def _get_param_names(self):
        return sum([[str(i)+'_'+k.name+'_'+n for n in k._get_param_names()] for i,k in enumerate(self.parts)],[])

    def _sort_slices(self,X,X2):
        slices = [index_to_slices(x) for x in X[:,-self.levels:].T]
        X = X[:,:-self.levels]
        if X2 is None:
            slices2 = slices
            X2 = X
        else:
            slices2 = [index_to_slices(x) for x in X2[:,-self.levels:].T]
            X2 = X2[:,:-self.levels]
        return X, X2, slices, slices2

    def K(self,X,X2,target):
        X, X2, slices, slices2 = self._sort_slices(X,X2)

        [[[[k.K(X[s],X2[s2],target[s,s2]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices_,slices2_)] for k, slices_, slices2_ in zip(self.parts,slices,slices2)]

    def Kdiag(self,X,target):
        raise NotImplementedError
        #X,slices = X[:,:-1],index_to_slices(X[:,-1])
        #[[self.k.Kdiag(X[s],target[s]) for s in slices_i] for slices_i in slices]

    def dK_dtheta(self,dL_dK,X,X2,target):
        X, X2, slices, slices2 = self._sort_slices(X,X2)
        [[[[k.dK_dtheta(dL_dK[s,s2],X[s],X2[s2],target[p_start:p_stop]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices_, slices2_)] for k, p_start, p_stop, slices_, slices2_ in zip(self.parts, self.param_starts, self.param_stops, slices, slices2)]


    def dK_dX(self,dL_dK,X,X2,target):
        raise NotImplementedError
        #X,slices = X[:,:-1],index_to_slices(X[:,-1])
        #if X2 is None:
            #X2,slices2 = X,slices
        #else:
            #X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
        #[[[self.k.dK_dX(dL_dK[s,s2],X[s],X2[s2],target[s,:-1]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
#
    def dKdiag_dX(self,dL_dKdiag,X,target):
        raise NotImplementedError
        #X,slices = X[:,:-1],index_to_slices(X[:,-1])
        #[[self.k.dKdiag_dX(dL_dKdiag[s],X[s],target[s,:-1]) for s in slices_i] for slices_i in slices]


    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        raise NotImplementedError
        #X,slices = X[:,:-1],index_to_slices(X[:,-1])
        #[[self.k.dKdiag_dX(dL_dKdiag[s],X[s],target) for s in slices_i] for slices_i in slices]
