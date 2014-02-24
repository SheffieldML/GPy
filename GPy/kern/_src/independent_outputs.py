# Copyright (c) 2012, James Hesnsman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern
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

class IndependentOutputs(Kern):
    """
    A kernel which can reopresent several independent functions.
    this kernel 'switches off' parts of the matrix where the output indexes are different.

    The index of the functions is given by the last column in the input X
    the rest of the columns of X are passed to the underlying kernel for computation (in blocks).

    """
    def __init__(self, kern, name='independ'):
        super(IndependentOutputs, self).__init__(kern.input_dim+1, name)
        self.kern = kern
        self.add_parameters(self.kern)

    def K(self,X ,X2=None):
        X, slices = X[:,:-1], index_to_slices(X[:,-1])
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0]))
            [[np.copyto(target[s,s], self.kern.K(X[s], None)) for s in slices_i] for slices_i in slices]
        else:
            X2, slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
            target = np.zeros((X.shape[0], X2.shape[0]))
            [[[np.copyto(target[s, s2], self.kern.K(X[s],X2[s2])) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
        return target

    def Kdiag(self,X):
        X, slices = X[:,:-1], index_to_slices(X[:,-1])
        target = np.zeros(X.shape[0])
        [[np.copyto(target[s], self.kern.Kdiag(X[s])) for s in slices_i] for slices_i in slices]
        return target

    def update_gradients_full(self,dL_dK,X,X2=None):
        target = np.zeros(self.kern.size)
        def collate_grads(dL, X, X2):
            self.kern.update_gradients_full(dL,X,X2)
            self.kern._collect_gradient(target)

        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            [[collate_grads(dL_dK[s,s], X[s], None) for s in slices_i] for slices_i in slices]
        else:
            X2, slices2 = X2[:,:-1], index_to_slices(X2[:,-1])
            [[[collate_grads(dL_dK[s,s2],X[s],X2[s2]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]

        self.kern._set_gradient(target)

    def gradients_X(self,dL_dK, X, X2=None):
        target = np.zeros_like(X)
        X, slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            [[np.copyto(target[s,:-1], self.kern.gradients_X(dL_dK[s,s],X[s],None)) for s in slices_i] for slices_i in slices]
        else:
            X2,slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
            [[[np.copyto(target[s,:-1], self.kern.gradients_X(dL_dK[s,s2], X[s], X2[s2])) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        X, slices = X[:,:-1], index_to_slices(X[:,-1])
        target = np.zeros(X.shape)
        [[np.copyto(target[s,:-1], self.kern.gradients_X_diag(dL_dKdiag[s],X[s])) for s in slices_i] for slices_i in slices]
        return target

    def update_gradients_diag(self,dL_dKdiag,X,target):
        target = np.zeros(self.kern.size)
        def collate_grads(dL, X):
            self.kern.update_gradients_diag(dL,X)
            self.kern._collect_gradient(target)
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        [[collate_grads(dL_dKdiag[s], X[s,:]) for s in slices_i] for slices_i in slices]
        self.kern._set_gradient(target)

def Hierarchical(kern_f, kern_g, name='hierarchy'):
    """
    A kernel which can reopresent a simple hierarchical model.

    See Hensman et al 2013, "Hierarchical Bayesian modelling of gene expression time
    series across irregularly sampled replicates and clusters"
    http://www.biomedcentral.com/1471-2105/14/252

    The index of the functions is given by the last column in the input X
    the rest of the columns of X are passed to the underlying kernel for computation (in blocks).

    """
    assert kern_f.input_dim == kern_g.input_dim
    return kern_f + IndependentOutputs(kern_g)

