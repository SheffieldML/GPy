"""
A new kernel
"""

import numpy as np
from kern import Kern,CombinationKernel
from .independent_outputs import index_to_slices
import itertools
from GPy.kern import Linear,RBF


class SplitKern(CombinationKernel):
    """
    A kernel which can represent several independent functions.  this kernel
    'switches off' parts of the matrix where the output indexes are different.

    The index of the functions is given by the last column in the input X the
    rest of the columns of X are passed to the underlying kernel for
    computation (in blocks).

    :param kernels: either a kernel, or list of kernels to work with. If it is
    a list of kernels the indices in the index_dim, index the kernels you gave!
    """
    def __init__(self, kernel, Xp, index_dim=-1, name='SplitKern'):
        assert isinstance(index_dim, int), "The index dimension must be an integer!"
        self.kern = kernel
        self.kern_cross = SplitKern_cross(kernel,Xp)
        super(SplitKern, self).__init__(kernels=[self.kern, self.kern_cross], extra_dims=[index_dim], name=name)
        self.index_dim = index_dim

    def K(self,X ,X2=None):
        slices = index_to_slices(X[:,self.index_dim])
        assert len(slices)<=2, 'The Split kernel only support two different indices'
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0]))
            # diagonal blocks
            [[target.__setitem__((s,ss), self.kern.K(X[s,:], X[ss,:])) for s,ss in itertools.product(slices_i, slices_i)] for slices_i in slices]
            if len(slices)>1:
                # cross blocks
                [target.__setitem__((s,ss), self.kern_cross.K(X[s,:], X[ss,:])) for s,ss in itertools.product(slices[0], slices[1])]
                # cross blocks
                [target.__setitem__((s,ss), self.kern_cross.K(X[s,:], X[ss,:])) for s,ss in itertools.product(slices[1], slices[0])]
        else:
            slices2 = index_to_slices(X2[:,self.index_dim])
            assert len(slices2)<=2, 'The Split kernel only support two different indices'
            target = np.zeros((X.shape[0], X2.shape[0]))
            # diagonal blocks
            [[target.__setitem__((s,s2), self.kern.K(X[s,:],X2[s2,:])) for s,s2 in itertools.product(slices[i], slices2[i])] for i in xrange(min(len(slices),len(slices)))]
            if len(slices)>1:
                [target.__setitem__((s,s2), self.kern_cross.K(X[s,:],X2[s2,:])) for s,s2 in itertools.product(slices[1], slices2[0])]
            if len(slices2)>1:
                [target.__setitem__((s,s2), self.kern_cross.K(X[s,:],X2[s2,:])) for s,s2 in itertools.product(slices[0], slices2[1])]                
        return target

    def Kdiag(self,X):
        return self.kern.Kdiag(X)

    def update_gradients_full(self,dL_dK,X,X2=None):
        slices = index_to_slices(X[:,self.index_dim])
        target = np.zeros(self.kern.size)

        def collate_grads(dL, X, X2, cross=False):
            if cross:
                self.kern_cross.update_gradients_full(dL,X,X2)
                target[:] += self.kern_cross.kern.gradient
            else:
                self.kern.update_gradients_full(dL,X,X2)
                target[:] += self.kern.gradient
    
        if X2 is None:
            [[collate_grads(dL_dK[s,ss], X[s], X[ss]) for s,ss in itertools.product(slices_i, slices_i)] for slices_i in slices]
            if len(slices)>1:
                [collate_grads(dL_dK[s,ss], X[s], X[ss], True) for s,ss in itertools.product(slices[0], slices[1])]
                [collate_grads(dL_dK[s,ss], X[s], X[ss], True) for s,ss in itertools.product(slices[1], slices[0])]
        else:
            slices2 = index_to_slices(X2[:,self.index_dim])
            [[collate_grads(dL_dK[s,s2],X[s],X2[s2]) for s,s2 in itertools.product(slices[i], slices2[i])] for i in xrange(min(len(slices),len(slices)))]
            if len(slices)>1:
                [collate_grads(dL_dK[s,ss], X[s], X2[s2], True) for s,s2 in itertools.product(slices[1], slices2[0])]
            if len(slices2)>1:
                [collate_grads(dL_dK[s,ss], X[s], X2[s2], True) for s,s2 in itertools.product(slices[0], slices2[1])]
        self.kern.gradient = target

    def update_gradients_diag(self, dL_dKdiag, X):
        self.kern.update_gradients_diag(self, dL_dKdiag, X)

class SplitKern_cross(Kern):

    def __init__(self, kernel, Xp, name='SplitKern_cross'):
        assert isinstance(kernel, Kern)
        self.kern = kernel
        self.Xp = Xp
        super(SplitKern_cross, self).__init__(input_dim=kernel.input_dim, active_dims=None, name=name)
        
    def K(self, X, X2=None):
        if X2 is None:
            return np.dot(self.kern.K(X,self.Xp),self.kern.K(self.Xp,X))/self.kern.K(self.Xp,self.Xp)
        else:
            return np.dot(self.kern.K(X,self.Xp),self.kern.K(self.Xp,X2))/self.kern.K(self.Xp,self.Xp)
        
    def Kdiag(self, X):
        return np.inner(self.kern.K(X,self.Xp),self.kern.K(self.Xp,X).T)/self.kern.K(self.Xp,self.Xp)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
                        
        k1 = self.kern.K(X,self.Xp)
        k2 = self.kern.K(self.Xp,X2)
        k3 = self.kern.K(self.Xp,self.Xp)
        dL_dk1 = np.einsum('ij,j->i',dL_dK,k2[0])/k3[0,0]
        dL_dk2 = np.einsum('ij,i->j',dL_dK,k1[:,0])/k3[0,0]
        dL_dk3 = np.einsum('ij,ij->',dL_dK,-np.dot(k1,k2)/(k3[0,0]*k3[0,0]))

        self.kern.update_gradients_full(dL_dk1[:,None],X,self.Xp)
        grad = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dk2[None,:],self.Xp,X2)
        grad += self.kern.gradient.copy()
        self.kern.update_gradients_full(np.array([[dL_dk3]]),self.Xp,self.Xp)
        grad += self.kern.gradient.copy()
        
        self.kern.gradient = grad

    def update_gradients_diag(self, dL_dKdiag, X):
        k1 = self.kern.K(X,self.Xp)
        k2 = self.kern.K(self.Xp,X)
        k3 = self.kern.K(self.Xp,self.Xp)
        dL_dk1 = dL_dKdiag*k2[0]/k3
        dL_dk2 = dL_dKdiag*k1[:,0]/k3
        dL_dk3 = -dL_dKdiag*(k1[:,0]*k2[0]).sum()/(k3*k3)
        
        self.kern.update_gradients_full(dL_dk1[:,None],X,self.Xp)
        grad1 = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dk2[None,:],self.Xp,X)
        grad2 = self.kern.gradient.copy()
        self.kern.update_gradients_full(np.array([[dL_dk3]]),self.Xp,self.Xp)
        grad3 = self.kern.gradient.copy()
        
        self.kern.gradient = grad1+grad2+grad3
        

