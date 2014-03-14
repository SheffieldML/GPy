# Copyright (c) 2012, James Hesnsman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kern import Kern, CombinationKernel
import numpy as np
import itertools

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

class IndependentOutputs(CombinationKernel):
    """
    A kernel which can represent several independent functions.
    this kernel 'switches off' parts of the matrix where the output indexes are different.

    The index of the functions is given by the last column in the input X
    the rest of the columns of X are passed to the underlying kernel for computation (in blocks).

    """
    def __init__(self, kern, index_dim=-1, name='independ'):
        assert isinstance(index_dim, int), "IndependentOutputs kernel is only defined with one input dimension being the indeces"
        super(IndependentOutputs, self).__init__(kernels=[kern], extra_dims=[index_dim], name=name)
        self.index_dim = index_dim
        self.kern = kern
        #self.add_parameters(self.kern)

    def K(self,X ,X2=None):
        slices = index_to_slices(X[:,self.index_dim])
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0]))
            [[np.copyto(target[s,ss], self.kern.K(X[s,:], X[ss,:])) for s,ss in itertools.product(slices_i, slices_i)] for slices_i in slices]
        else:
            slices2 = index_to_slices(X2[:,self.index_dim])
            target = np.zeros((X.shape[0], X2.shape[0]))
            [[[np.copyto(target[s, s2], self.kern.K(X[s,:],X2[s2,:])) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
        return target

    def Kdiag(self,X):
        slices = index_to_slices(X[:,self.index_dim])
        target = np.zeros(X.shape[0])
        [[np.copyto(target[s], self.kern.Kdiag(X[s])) for s in slices_i] for slices_i in slices]
        return target

    def update_gradients_full(self,dL_dK,X,X2=None):
        target = np.zeros(self.kern.size)
        def collate_grads(dL, X, X2):
            self.kern.update_gradients_full(dL,X,X2)
            target[:] += self.kern.gradient

        slices = index_to_slices(X[:,self.index_dim])
        if X2 is None:
            [[collate_grads(dL_dK[s,ss], X[s], X[ss]) for s,ss in itertools.product(slices_i, slices_i)] for slices_i in slices]
        else:
            slices2 = index_to_slices(X2[:,self.index_dim])
            [[[collate_grads(dL_dK[s,s2],X[s],X2[s2]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
        self.kern.gradient = target

    def gradients_X(self,dL_dK, X, X2=None):
        target = np.zeros_like(X)
        slices = index_to_slices(X[:,self.index_dim])
        if X2 is None:
            [[np.copyto(target[s,self.kern.active_dims], self.kern.gradients_X(dL_dK[s,s],X[s],X[ss])) for s, ss in product(slices_i, slices_i)] for slices_i in slices]
        else:
            X2,slices2 = X2[:,:self.index_dim],index_to_slices(X2[:,-1])
            [[[np.copyto(target[s,:self.index_dim], self.kern.gradients_X(dL_dK[s,s2], X[s], X2[s2])) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        slices = index_to_slices(X[:,self.index_dim])
        target = np.zeros(X.shape)
        [[np.copyto(target[s,:-1], self.kern.gradients_X_diag(dL_dKdiag[s],X[s])) for s in slices_i] for slices_i in slices]
        return target

    def update_gradients_diag(self, dL_dKdiag, X):
        target = np.zeros(self.kern.size)
        def collate_grads(dL, X):
            self.kern.update_gradients_diag(dL,X)
            target[:] += self.kern.gradient
        slices = index_to_slices(X[:,self.index_dim])
        [[collate_grads(dL_dKdiag[s], X[s,:]) for s in slices_i] for slices_i in slices]
        self.kern.gradient = target

class Hierarchical(Kern):
    """
    A kernel which can reopresent a simple hierarchical model.

    See Hensman et al 2013, "Hierarchical Bayesian modelling of gene expression time
    series across irregularly sampled replicates and clusters"
    http://www.biomedcentral.com/1471-2105/14/252

    The index of the functions is given by additional columns in the input X.

    """
    def __init__(self, kerns, name='hierarchy'):
        assert all([k.input_dim==kerns[0].input_dim for k in kerns])
        super(Hierarchical, self).__init__(kerns[0].input_dim + len(kerns) - 1, name)
        self.kerns = kerns
        self.add_parameters(self.kerns)

    def K(self,X ,X2=None):
        X, slices = X[:,:-self.levels], [index_to_slices(X[:,i]) for i in range(self.kerns[0].input_dim, self.input_dim)]
        K = self.kerns[0].K(X, X2)
        if X2 is None:
            [[[np.copyto(K[s,s], k.K(X[s], None)) for s in slices_i] for slices_i in slices_k] for k, slices_k in zip(self.kerns[1:], slices)]
        else:
            X2, slices2 = X2[:,:-1],index_to_slices(X2[:,-1])
            [[[[np.copyto(K[s, s2], self.kern.K(X[s],X2[s2])) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices_k,slices_k2)] for k, slices_k, slices_k2 in zip(self.kerns[1:], slices, slices2)]
        return target

    def Kdiag(self,X):
        X, slices = X[:,:-self.levels], [index_to_slices(X[:,i]) for i in range(self.kerns[0].input_dim, self.input_dim)]
        K = self.kerns[0].K(X, X2)
        [[[np.copyto(target[s], self.kern.Kdiag(X[s])) for s in slices_i] for slices_i in slices_k] for k, slices_k in zip(self.kerns[1:], slices)]
        return target

    def update_gradients_full(self,dL_dK,X,X2=None):
        X,slices = X[:,:-1],index_to_slices(X[:,-1])
        if X2 is None:
            self.kerns[0].update_gradients_full(dL_dK, X, None)
            for k, slices_k in zip(self.kerns[1:], slices):
                target = np.zeros(k.size)
                def collate_grads(dL, X, X2):
                    k.update_gradients_full(dL,X,X2)
                    k._collect_gradient(target)
                [[k.update_gradients_full(dL_dK[s,s], X[s], None) for s in slices_i] for slices_i in slices_k]
                k._set_gradient(target)
        else:
            X2, slices2 = X2[:,:-1], index_to_slices(X2[:,-1])
            self.kerns[0].update_gradients_full(dL_dK, X, None)
            for k, slices_k in zip(self.kerns[1:], slices):
                target = np.zeros(k.size)
                def collate_grads(dL, X, X2):
                    k.update_gradients_full(dL,X,X2)
                    k._collect_gradient(target)
                [[[collate_grads(dL_dK[s,s2],X[s],X2[s2]) for s in slices_i] for s2 in slices_j] for slices_i,slices_j in zip(slices,slices2)]
                k._set_gradient(target)


