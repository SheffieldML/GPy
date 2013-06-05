# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
import hashlib
#from scipy import integrate # This may not be necessary (Nicolas, 20th Feb)

class prod_orthogonal(Kernpart):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kernpart
    :rtype: kernel object

    """
    def __init__(self,k1,k2):
        self.input_dim = k1.input_dim + k2.input_dim
        self.num_params = k1.num_params + k2.num_params
        self.name = k1.name + '<times>' + k2.name
        self.k1 = k1
        self.k2 = k2
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

    def dK_dtheta(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to the parameters."""
        self._K_computations(X,X2)
        if X2 is None:
            self.k1.dK_dtheta(dL_dK*self._K2, X[:,:self.k1.input_dim], None, target[:self.k1.num_params])
            self.k2.dK_dtheta(dL_dK*self._K1, X[:,self.k1.input_dim:], None, target[self.k1.num_params:])
        else:
            self.k1.dK_dtheta(dL_dK*self._K2, X[:,:self.k1.input_dim], X2[:,:self.k1.input_dim], target[:self.k1.num_params])
            self.k2.dK_dtheta(dL_dK*self._K1, X[:,self.k1.input_dim:], X2[:,self.k1.input_dim:], target[self.k1.num_params:])

    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        target1 = np.zeros(X.shape[0])
        target2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,:self.k1.input_dim],target1)
        self.k2.Kdiag(X[:,self.k1.input_dim:],target2)
        target += target1 * target2

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        K1 = np.zeros(X.shape[0])
        K2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,:self.k1.input_dim],K1)
        self.k2.Kdiag(X[:,self.k1.input_dim:],K2)
        self.k1.dKdiag_dtheta(dL_dKdiag*K2,X[:,:self.k1.input_dim],target[:self.k1.num_params])
        self.k2.dKdiag_dtheta(dL_dKdiag*K1,X[:,self.k1.input_dim:],target[self.k1.num_params:])

    def dK_dX(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to X."""
        self._K_computations(X,X2)
        self.k1.dK_dX(dL_dK*self._K2, X[:,:self.k1.input_dim], X2[:,:self.k1.input_dim], target)
        self.k2.dK_dX(dL_dK*self._K1, X[:,self.k1.input_dim:], X2[:,self.k1.input_dim:], target)

    def dKdiag_dX(self, dL_dKdiag, X, target):
        K1 = np.zeros(X.shape[0])
        K2 = np.zeros(X.shape[0])
        self.k1.Kdiag(X[:,0:self.k1.input_dim],K1)
        self.k2.Kdiag(X[:,self.k1.input_dim:],K2)

        self.k1.dK_dX(dL_dKdiag*K2, X[:,:self.k1.input_dim], target)
        self.k2.dK_dX(dL_dKdiag*K1, X[:,self.k1.input_dim:], target)

    def _K_computations(self,X,X2):
        if not (np.array_equal(X,self._X) and np.array_equal(X2,self._X2) and np.array_equal(self._params , self._get_params())):
            self._X = X.copy()
            self._params == self._get_params().copy()
            if X2 is None:
                self._X2 = None
                self._K1 = np.zeros((X.shape[0],X.shape[0]))
                self._K2 = np.zeros((X.shape[0],X.shape[0]))
                self.k1.K(X[:,:self.k1.input_dim],None,self._K1)
                self.k2.K(X[:,self.k1.input_dim:],None,self._K2)
            else:
                self._X2 = X2.copy()
                self._K1 = np.zeros((X.shape[0],X2.shape[0]))
                self._K2 = np.zeros((X.shape[0],X2.shape[0]))
                self.k1.K(X[:,:self.k1.input_dim],X2[:,:self.k1.input_dim],self._K1)
                self.k2.K(X[:,self.k1.input_dim:],X2[:,self.k1.input_dim:],self._K2)

