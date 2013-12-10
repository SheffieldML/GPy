# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
from coregionalize import Coregionalize
import numpy as np
import hashlib

class Prod(Kernpart):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kernpart
    :param tensor: The kernels are either multiply as functions defined on the same input space (default) or on the product of the input spaces
    :type tensor: Boolean
    :rtype: kernel object

    """
    def __init__(self,k1,k2,tensor=False):
        self.num_params = k1.num_params + k2.num_params
        if tensor:
            self.name = '['+k1.name + '**' + k2.name +']'
        else:
            self.name = '['+k1.name + '*' + k2.name +']'
        self.k1 = k1
        self.k2 = k2
        if tensor:
            self.input_dim = k1.input_dim + k2.input_dim
            self.slice1 = slice(0,self.k1.input_dim)
            self.slice2 = slice(self.k1.input_dim,self.k1.input_dim+self.k2.input_dim)
        else:
            assert k1.input_dim == k2.input_dim, "Error: The input spaces of the kernels to sum don't have the same dimension."
            self.input_dim = k1.input_dim
            self.slice1 = slice(0,self.input_dim)
            self.slice2 = slice(0,self.input_dim)

        self._X, self._X2, self._params = np.empty(shape=(3,1))
        self._set_params(np.hstack((k1._get_params(),k2._get_params())))

    def _get_params(self):
        """return the value of the parameters."""
        return np.hstack((self.k1._get_params(), self.k2._get_params()))

    def _set_params(self,x):
        """set the value of the parameters."""
        self.k1._set_params(x[:self.k1.num_params])
        self.k2._set_params(x[self.k1.num_params:])

    def _get_param_names(self):
        """return parameter names."""
        return [self.k1.name + '_' + param_name for param_name in self.k1._get_param_names()] + [self.k2.name + '_' + param_name for param_name in self.k2._get_param_names()]

    def K(self,X,X2,target):
        self._K_computations(X,X2)
        target += self._K1 * self._K2

    def K1(self,X, X2):
        """Compute the part of the kernel associated with k1."""
        self._K_computations(X, X2)
        return self._K1

    def K2(self, X, X2):
        """Compute the part of the kernel associated with k2."""
        self._K_computations(X, X2)
        return self._K2

    def dK_dtheta(self,dL_dK,X,X2,target):
        """Derivative of the covariance matrix with respect to the parameters."""
        self._K_computations(X,X2)
        if X2 is None:
            self.k1.dK_dtheta(dL_dK*self._K2, X[:,self.slice1], None, target[:self.k1.num_params])
            self.k2.dK_dtheta(dL_dK*self._K1, X[:,self.slice2], None, target[self.k1.num_params:])
        else:
            self.k1.dK_dtheta(dL_dK*self._K2, X[:,self.slice1], X2[:,self.slice1], target[:self.k1.num_params])
            self.k2.dK_dtheta(dL_dK*self._K1, X[:,self.slice2], X2[:,self.slice2], target[self.k1.num_params:])

    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        target1 = np.zeros(X.shape[0])
        target2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,self.slice1],target1)
        self.k2.Kdiag(X[:,self.slice2],target2)
        target += target1 * target2

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        K1 = np.zeros(X.shape[0])
        K2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,self.slice1],K1)
        self.k2.Kdiag(X[:,self.slice2],K2)
        self.k1.dKdiag_dtheta(dL_dKdiag*K2,X[:,self.slice1],target[:self.k1.num_params])
        self.k2.dKdiag_dtheta(dL_dKdiag*K1,X[:,self.slice2],target[self.k1.num_params:])

    def dK_dX(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to X."""
        self._K_computations(X,X2)
        if X2 is None:
            if not isinstance(self.k1,Coregionalize) and not isinstance(self.k2,Coregionalize):
                self.k1.dK_dX(dL_dK*self._K2, X[:,self.slice1], None, target[:,self.slice1])
                self.k2.dK_dX(dL_dK*self._K1, X[:,self.slice2], None, target[:,self.slice2])
            else:#if isinstance(self.k1,Coregionalize) or isinstance(self.k2,Coregionalize):
                #NOTE The indices column in the inputs makes the ki.dK_dX fail when passing None instead of X[:,self.slicei]
                X2 = X
                self.k1.dK_dX(2.*dL_dK*self._K2, X[:,self.slice1], X2[:,self.slice1], target[:,self.slice1])
                self.k2.dK_dX(2.*dL_dK*self._K1, X[:,self.slice2], X2[:,self.slice2], target[:,self.slice2])
        else:
            self.k1.dK_dX(dL_dK*self._K2, X[:,self.slice1], X2[:,self.slice1], target[:,self.slice1])
            self.k2.dK_dX(dL_dK*self._K1, X[:,self.slice2], X2[:,self.slice2], target[:,self.slice2])

    def dKdiag_dX(self, dL_dKdiag, X, target):
        K1 = np.zeros(X.shape[0])
        K2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,self.slice1],K1)
        self.k2.Kdiag(X[:,self.slice2],K2)

        self.k1.dK_dX(dL_dKdiag*K2, X[:,self.slice1], target[:,self.slice1])
        self.k2.dK_dX(dL_dKdiag*K1, X[:,self.slice2], target[:,self.slice2])

    def _K_computations(self,X,X2):
        if not (np.array_equal(X,self._X) and np.array_equal(X2,self._X2) and np.array_equal(self._params , self._get_params())):
            self._X = X.copy()
            self._params == self._get_params().copy()
            if X2 is None:
                self._X2 = None
                self._K1 = np.zeros((X.shape[0],X.shape[0]))
                self._K2 = np.zeros((X.shape[0],X.shape[0]))
                self.k1.K(X[:,self.slice1],None,self._K1)
                self.k2.K(X[:,self.slice2],None,self._K2)
            else:
                self._X2 = X2.copy()
                self._K1 = np.zeros((X.shape[0],X2.shape[0]))
                self._K2 = np.zeros((X.shape[0],X2.shape[0]))
                self.k1.K(X[:,self.slice1],X2[:,self.slice1],self._K1)
                self.k2.K(X[:,self.slice2],X2[:,self.slice2],self._K2)

    #def __getstate__(self):
        #return [self.k1, self.k2, self.slice1, self.slice2, self.name, self.input_dim, self.num_params]

    #def __setstate__(self, state):
        #self.k1, self.k2, self.slice1, self.slice2, self.name, self.input_dim, self.num_params = state
        #self._X, self._X2, self._params = np.empty(shape=(3,1))




