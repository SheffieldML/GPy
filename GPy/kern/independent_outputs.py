# Copyright (c) 2012, James Hesnsman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np

def index_to_slices(index):
    """
    take a numpy array of integers (index) and return a  nested list of slices such that the slices describe the start, stop points for each integer in the index. 

    e.g.
    >>> index = np.asarray([0,0,0,1,1,1,2,2,2])
    returns
    >>> [[slice(0,3,None)],[slice(3,6,None)],[slice(6,9,None)]]

    or, a more complicated example
    >>> index = np.asarray([0,0,1,1,0,2,2,2,1,1])
    returns
    >>> [[slice(0,2,None),slice(4,5,None)],[slice(2,4,None),slice(8,10,None)],[slice(5,8,None)]]
    """

    #contruct the return structure
    ind = np.asarray(index,dtype=np.int64)
    ret = [[] for i in range(ind.max()+1)]

    #find the switchpoints
    ind_ = np.hstack((ind,ind[0]+ind[-1]+1))
    switchpoints = np.nonzero(ind_ - np.roll(ind_,+1))[0]

    [ret[ind_i].append(slice(*indexes_i)) for ind_i,indexes_i in zip(ind[switchpoints[:-1]],zip(switchpoints,switchpoints[1:]))]
    return ret

class IndependentOutputs(Kernpart):
    """
    A kernel part shich can reopresent several independent functions.
    this kernel 'switches off' parts of the matrix where the output indexes are different.

    The index of the functions is given by the last column in the input X
    the rest of the columns of X are passed to the kernel for computation (in blocks).

    """
    def __init__(self,k):
        self.input_dim = k.input_dim + 1
        self.num_params = k.num_params
        self.name = 'iops('+ k.name + ')'
        self.k = k

    def _get_params(self):
        return self.k._get_params()

    def _set_params(self,x):
        self.k._set_params(x)
        self.params = x

    def _get_param_names(self):
        return self.k._get_param_names()

    def K(self,X,X2,target):
        #Sort out the slices from the input data
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])

        [[[self.k.K(X[s],X2[s2],target[s,s2]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]

    def Kdiag(self,X,target):
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        [[self.k.Kdiag(X[s],target[s]) for s in slices_i] for slices_i in slices]

    def dK_dtheta(self,dL_dK,X,X2,target):
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
        [[[self.k.dK_dtheta(dL_dK[s,s2],X[s],X2[s2],target) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]


    def dK_dX(self,dL_dK,X,X2,target):
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            X2,slices2 = X,slices
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
        [[[self.k.dK_dX(dL_dK[s,s2],X[s],X2[s2],target[s,:-1]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]

    def dKdiag_dX(self,dL_dKdiag,X,target):
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        [[self.k.dKdiag_dX(dL_dKdiag[s],X[s],target[s,:-1]) for s in slices_i] for slices_i in slices]


    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        [[self.k.dKdiag_dX(dL_dKdiag[s],X[s],target) for s in slices_i] for slices_i in slices]
