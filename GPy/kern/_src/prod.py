# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kern import Kern
import numpy as np

class Prod(Kern):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kern
    :param tensor: The kernels are either multiply as functions defined on the same input space (default) or on the product of the input spaces
    :type tensor: Boolean
    :rtype: kernel object

    """
    def __init__(self, k1, k2, tensor=False,name=None):
        if tensor:
            name = k1.name + '_xx_' + k2.name if name is None else name
            super(Prod, self).__init__(k1.input_dim + k2.input_dim, name)
            self.slice1 = slice(0,k1.input_dim)
            self.slice2 = slice(k1.input_dim,k1.input_dim+k2.input_dim)
        else:
            assert k1.input_dim == k2.input_dim, "Error: The input spaces of the kernels to multiply don't have the same dimension."
            name = k1.name + '_x_' + k2.name if name is None else name
            super(Prod, self).__init__(k1.input_dim, name)
            self.slice1 = slice(0, self.input_dim)
            self.slice2 = slice(0, self.input_dim)
        self.k1 = k1
        self.k2 = k2
        self.add_parameters(self.k1, self.k2)

    def K(self, X, X2=None):
        if X2 is None:
            return self.k1.K(X[:,self.slice1], None) * self.k2.K(X[:,self.slice2], None)
        else:
            return self.k1.K(X[:,self.slice1], X2[:,self.slice1]) * self.k2.K(X[:,self.slice2], X2[:,self.slice2])

    def Kdiag(self, X):
        return self.k1.Kdiag(X[:,self.slice1]) * self.k2.Kdiag(X[:,self.slice2])

    def update_gradients_full(self, dL_dK, X):
        self.k1.update_gradients_full(dL_dK*self.k2.K(X[:,self.slice2]), X[:,self.slice1])
        self.k2.update_gradients_full(dL_dK*self.k1.K(X[:,self.slice1]), X[:,self.slice2])

    def gradients_X(self, dL_dK, X, X2=None):
        target = np.zeros(X.shape)
        if X2 is None:
            target[:,self.slice1] += self.k1.gradients_X(dL_dK*self.k2.K(X[:,self.slice2]), X[:,self.slice1], None)
            target[:,self.slice2] += self.k2.gradients_X(dL_dK*self.k1.K(X[:,self.slice1]), X[:,self.slice2], None)
        else:
            target[:,self.slice1] += self.k1.gradients_X(dL_dK*self.k2.K(X[:,self.slice2], X2[:,self.slice2]), X[:,self.slice1], X2[:,self.slice1])
            target[:,self.slice2] += self.k2.gradients_X(dL_dK*self.k1.K(X[:,self.slice1], X2[:,self.slice1]), X[:,self.slice2], X2[:,self.slice2])
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape)
        target[:,self.slice1] = self.k1.gradients_X(dL_dKdiag*self.k2.Kdiag(X[:,self.slice2]), X[:,self.slice1])
        target[:,self.slice2] += self.k2.gradients_X(dL_dKdiag*self.k1.Kdiag(X[:,self.slice1]), X[:,self.slice2])
        return target


