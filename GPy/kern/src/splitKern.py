"""
A new kernel
"""

import numpy as np
from .kern import Kern, CombinationKernel
from ...util.multioutput import index_to_slices
import itertools

class DEtime(Kern):

    def __init__(self, kernel, idx_p, Xp, index_dim=-1, name='DiffGenomeKern'):
        self.idx_p = idx_p
        self.index_dim=index_dim
        self.kern = SplitKern(kernel,Xp, index_dim=index_dim)
        super(DEtime, self).__init__(input_dim=kernel.input_dim+1, active_dims=None, name=name)
        self.add_parameter(self.kern)
    
    def K(self, X, X2=None):
        assert X2==None
        K = self.kern.K(X,X2)
        
        if self.idx_p<=0 or self.idx_p>X.shape[0]/2:
            return K
        
        slices = index_to_slices(X[:,self.index_dim])
        idx_start = slices[1][0].start
        idx_end = idx_start+self.idx_p
        K_c = K[idx_start:idx_end,idx_start:idx_end].copy()
        K[idx_start:idx_end,:] = K[:self.idx_p,:]
        K[:,idx_start:idx_end] = K[:,:self.idx_p]
        K[idx_start:idx_end,idx_start:idx_end] = K_c
        
        return K
    
    def Kdiag(self,X):
        Kdiag = self.kern.Kdiag(X)

        if self.idx_p<=0 or self.idx_p>X.shape[0]/2:
            return Kdiag

        slices = index_to_slices(X[:,self.index_dim])
        idx_start = slices[1][0].start
        idx_end = idx_start+self.idx_p
        Kdiag[idx_start:idx_end] = Kdiag[:self.idx_p]
        
        return Kdiag
    
    def update_gradients_full(self,dL_dK,X,X2=None):
        assert X2==None
        if self.idx_p<=0 or self.idx_p>X.shape[0]/2:
            self.kern.update_gradients_full(dL_dK, X)
            return
        
        slices = index_to_slices(X[:,self.index_dim])
        idx_start = slices[1][0].start
        idx_end = idx_start+self.idx_p
        
        self.kern.update_gradients_full(dL_dK[idx_start:idx_end,:], X[:self.idx_p],X)
        grad_p1 = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dK[:,idx_start:idx_end], X, X[:self.idx_p])
        grad_p2 = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dK[idx_start:idx_end,idx_start:idx_end], X[:self.idx_p],X[idx_start:idx_end])
        grad_p3 = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dK[idx_start:idx_end,idx_start:idx_end], X[idx_start:idx_end], X[:self.idx_p])
        grad_p4 = self.kern.gradient.copy()

        self.kern.update_gradients_full(dL_dK[idx_start:idx_end,:], X[idx_start:idx_end],X)
        grad_n1 = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dK[:,idx_start:idx_end], X, X[idx_start:idx_end])
        grad_n2 = self.kern.gradient.copy()
        self.kern.update_gradients_full(dL_dK[idx_start:idx_end,idx_start:idx_end], X[idx_start:idx_end], X[idx_start:idx_end])
        grad_n3 = self.kern.gradient.copy()

        self.kern.update_gradients_full(dL_dK, X)
        self.kern.gradient += grad_p1+grad_p2-grad_p3-grad_p4-grad_n1-grad_n2+2*grad_n3

    def update_gradients_diag(self, dL_dKdiag, X):
        pass

class SplitKern(CombinationKernel):

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
            [[target.__setitem__((s,s2), self.kern.K(X[s,:],X2[s2,:])) for s,s2 in itertools.product(slices[i], slices2[i])] for i in range(min(len(slices),len(slices2)))]
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
            assert dL_dK.shape==(X.shape[0],X.shape[0])
            [[collate_grads(dL_dK[s,ss], X[s], X[ss]) for s,ss in itertools.product(slices_i, slices_i)] for slices_i in slices]
            if len(slices)>1:
                [collate_grads(dL_dK[s,ss], X[s], X[ss], True) for s,ss in itertools.product(slices[0], slices[1])]
                [collate_grads(dL_dK[s,ss], X[s], X[ss], True) for s,ss in itertools.product(slices[1], slices[0])]
        else:
            assert dL_dK.shape==(X.shape[0],X2.shape[0])
            slices2 = index_to_slices(X2[:,self.index_dim])
            [[collate_grads(dL_dK[s,s2],X[s],X2[s2]) for s,s2 in itertools.product(slices[i], slices2[i])] for i in range(min(len(slices),len(slices2)))]
            if len(slices)>1:
                [collate_grads(dL_dK[s,s2], X[s], X2[s2], True) for s,s2 in itertools.product(slices[1], slices2[0])]
            if len(slices2)>1:
                [collate_grads(dL_dK[s,s2], X[s], X2[s2], True) for s,s2 in itertools.product(slices[0], slices2[1])]
        self.kern.gradient = target

    def update_gradients_diag(self, dL_dKdiag, X):
        self.kern.update_gradients_diag(self, dL_dKdiag, X)

class SplitKern_cross(Kern):

    def __init__(self, kernel, Xp, name='SplitKern_cross'):
        assert isinstance(kernel, Kern)
        self.kern = kernel
        if not isinstance(Xp,np.ndarray):
            Xp = np.array([[Xp]])
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
        

