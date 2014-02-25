# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
from ...util.linalg import pdinv,mdot

class FiniteDimensional(Kernpart):
    def __init__(self, input_dim, F, G, variance=1., weights=None):
        """
        Argumnents
        ----------
        input_dim: int - the number of input dimensions
        F: np.array of functions with shape (n,) - the n basis functions
        G: np.array with shape (n,n) - the Gram matrix associated to F
        weights : np.ndarray with shape (n,)
        """
        self.input_dim = input_dim
        self.F = F
        self.G = G
        self.G_1 ,L,Li,logdet = pdinv(G)
        self.n = F.shape[0]
        if weights is not None:
            assert weights.shape==(self.n,)
        else:
            weights = np.ones(self.n)
        self.num_params = self.n + 1
        self.name = 'finite_dim'
        self._set_params(np.hstack((variance,weights)))

    def _get_params(self):
        return np.hstack((self.variance,self.weights))
    def _set_params(self,x):
        assert x.size == (self.num_params)
        self.variance = x[0]
        self.weights = x[1:]
    def _get_param_names(self):
        if self.n==1:
            return ['variance','weight']
        else:
            return ['variance']+['weight_%i'%i for i in range(self.weights.size)]

    def K(self,X,X2,target):
        if X2 is None: X2 = X
        FX = np.column_stack([f(X) for f in self.F])
        FX2 = np.column_stack([f(X2) for f in self.F])
        product = self.variance * mdot(FX,np.diag(np.sqrt(self.weights)),self.G_1,np.diag(np.sqrt(self.weights)),FX2.T)
        np.add(product,target,target)
    def Kdiag(self,X,target):
        product = np.diag(self.K(X, X))
        np.add(target,product,target)
    def _param_grad_helper(self,X,X2,target):
        """Return shape is NxMx(Ntheta)"""
        if X2 is None: X2 = X
        FX = np.column_stack([f(X) for f in self.F])
        FX2 = np.column_stack([f(X2) for f in self.F])
        DER = np.zeros((self.n,self.n,self.n))
        for i in range(self.n):
            DER[i,i,i] = np.sqrt(self.weights[i])
        dw = self.variance * mdot(FX,DER,self.G_1,np.diag(np.sqrt(self.weights)),FX2.T)
        dv = mdot(FX,np.diag(np.sqrt(self.weights)),self.G_1,np.diag(np.sqrt(self.weights)),FX2.T)
        np.add(target[:,:,0],np.transpose(dv,(0,2,1)), target[:,:,0])
        np.add(target[:,:,1:],np.transpose(dw,(0,2,1)), target[:,:,1:])
    def dKdiag_dtheta(self,X,target):
        np.add(target[:,0],1.,target[:,0])








