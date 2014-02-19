# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kern import Kern
from coregionalize import Coregionalize
import numpy as np
import hashlib

class Prod(Kern):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kern
    :param tensor: The kernels are either multiply as functions defined on the same input space (default) or on the product of the input spaces
    :type tensor: Boolean
    :rtype: kernel object

    """
    def __init__(self,k1,k2,tensor=False):
        if tensor:
            super(Prod, self).__init__(k1.input_dim + k2.input_dim, k1.name + '_xx_' + k2.name)
            self.slice1 = slice(0,k1.input_dim)
            self.slice2 = slice(k1.input_dim,k1.input_dim+k2.input_dim)
        else:
            assert k1.input_dim == k2.input_dim, "Error: The input spaces of the kernels to multiply don't have the same dimension."
            super(Prod, self).__init__(k1.input_dim, k1.name + '_x_' + k2.name)
            self.slice1 = slice(0,self.input_dim)
            self.slice2 = slice(0,self.input_dim)
        self.k1 = k1
        self.k2 = k2
        self.add_parameters(self.k1, self.k2)

        #initialize cache
        self._X, self._X2 = np.empty(shape=(2,1))
        self._params = None

    def K(self, X, X2=None):
        self._K_computations(X, X2)
        return self._K1 * self._K2

    def Kdiag(self, X):
        return self.k1.Kdiag(X[:,self.slice1]) * self.k2.Kdiag(X[:,self.slice2])

    def update_gradients_full(self, dL_dK, X):
        self._K_computations(X, None)
        self.k1.update_gradients_full(dL_dK*self._K2, X[:,self.slice1])
        self.k2.update_gradients_full(dL_dK*self._K1, X[:,self.slice2])

    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        self.k1.update_gradients_sparse(dL_dKmm * self.k2.K(Z[:,self.slice2]), dL_dKnm * self.k2(X[:,self.slice2], Z[:,self.slice2]), dL_dKdiag * self.k2.Kdiag(X[:,self.slice2]), X[:,self.slice1], Z[:,self.slice1] )
        self.k2.update_gradients_sparse(dL_dKmm * self.k1.K(Z[:,self.slice1]), dL_dKnm * self.k1(X[:,self.slice1], Z[:,self.slice1]), dL_dKdiag * self.k1.Kdiag(X[:,self.slice1]), X[:,self.slice2], Z[:,self.slice2] )

    def gradients_X(self, dL_dK, X, X2=None):
        """derivative of the covariance matrix with respect to X."""
        self._K_computations(X, X2)
        target = np.zeros(X.shape)
        if X2 is None:
            target[:,self.slice1] += self.k1.gradients_X(dL_dK*self._K2, X[:,self.slice1], None)
            target[:,self.slice2] += self.k2.gradients_X(dL_dK*self._K1, X[:,self.slice2], None)
        else:
            target[:,self.slice1] += self.k1.gradients_X(dL_dK*self._K2, X[:,self.slice1], X2[:,self.slice1])
            target[:,self.slice2] += self.k2.gradients_X(dL_dK*self._K1, X[:,self.slice2], X2[:,self.slice2])

        return target

    def dKdiag_dX(self, dL_dKdiag, X, target):
        K1 = np.zeros(X.shape[0])
        K2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,self.slice1],K1)
        self.k2.Kdiag(X[:,self.slice2],K2)

        self.k1.gradients_X(dL_dKdiag*K2, X[:,self.slice1], target[:,self.slice1])
        self.k2.gradients_X(dL_dKdiag*K1, X[:,self.slice2], target[:,self.slice2])

    def _K_computations(self, X, X2):
        if not (np.array_equal(X,self._X) and np.array_equal(X2,self._X2) and np.array_equal(self._params , self._get_params())):
            self._X = X.copy()
            self._params == self._get_params().copy()
            if X2 is None:
                self._X2 = None
                self._K1 = self.k1.K(X[:,self.slice1],None)
                self._K2 = self.k2.K(X[:,self.slice2],None)
            else:
                self._X2 = X2.copy()
                self._K1 = self.k1.K(X[:,self.slice1],X2[:,self.slice1])
                self._K2 = self.k2.K(X[:,self.slice2],X2[:,self.slice2])

