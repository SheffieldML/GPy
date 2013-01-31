# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import kernpart
import numpy as np
import hashlib
from scipy import integrate

class product_orthogonal(kernpart):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: kernpart
    :rtype: kernel object

    """
    def __init__(self,k1,k2):
        assert k1._get_param_names()[0] == 'variance' and k2._get_param_names()[0] == 'variance', "Error: The multipication of kernels is only defined when the first parameters of the kernels to multiply is the variance."
        self.D = k1.D + k2.D
        self.Nparam = k1.Nparam + k2.Nparam - 1
        self.name = k1.name + '<times>' + k2.name
        self.k1 = k1
        self.k2 = k2
        self._set_params(np.hstack((k1._get_params()[0]*k2._get_params()[0], k1._get_params()[1:],k2._get_params()[1:])))

    def _get_params(self):
        """return the value of the parameters."""
        return self.params

    def _set_params(self,x):
        """set the value of the parameters."""
        self.k1._set_params(np.hstack((1.,x[1:self.k1.Nparam])))
        self.k2._set_params(np.hstack((1.,x[self.k1.Nparam:])))
        self.params = x

    def _get_param_names(self):
        """return parameter names."""
        return ['variance']+[self.k1.name + '_' + self.k1._get_param_names()[i+1] for i in range(self.k1.Nparam-1)] +  [self.k2.name + '_' + self.k2._get_param_names()[i+1] for i in range(self.k2.Nparam-1)]

    def K(self,X,X2,target):
        """Compute the covariance matrix between X and X2."""
        if X2 is None: X2 = X
        target1 = np.zeros((X.shape[0],X2.shape[0]))
        target2 = np.zeros((X.shape[0],X2.shape[0]))
        self.k1.K(X[:,0:self.k1.D],X2[:,0:self.k1.D],target1)
        self.k2.K(X[:,self.k1.D:],X2[:,self.k1.D:],target2)
        target += self.params[0]*target1 * target2

    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        target1 = np.zeros((X.shape[0],))
        target2 = np.zeros((X.shape[0],))
        self.k1.Kdiag(X[:,0:self.k1.D],target1)
        self.k2.Kdiag(X[:,self.k1.D:],target2)
        target += self.params[0]*target1 * target2

    def dK_dtheta(self,partial,X,X2,target):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None: X2 = X
        K1 = np.zeros((X.shape[0],X2.shape[0]))
        K2 = np.zeros((X.shape[0],X2.shape[0]))
        self.k1.K(X[:,0:self.k1.D],X2[:,0:self.k1.D],K1)
        self.k2.K(X[:,self.k1.D:],X2[:,self.k1.D:],K2)

        k1_target = np.zeros(self.k1.Nparam)
        k2_target = np.zeros(self.k2.Nparam)
        self.k1.dK_dtheta(partial*K2, X[:,:self.k1.D], X2[:,:self.k1.D], k1_target)
        self.k2.dK_dtheta(partial*K1, X[:,self.k1.D:], X2[:,self.k1.D:], k2_target)

        target[0] += np.sum(K1*K2*partial)
        target[1:self.k1.Nparam] += self.params[0]* k1_target[1:]
        target[self.k1.Nparam:] += self.params[0]* k2_target[1:]
        
    def dKdiag_dtheta(self,partial,X,target):
        """derivative of the diagonal of the covariance matrix with respect to the parameters."""
        target[0] += 1

    def dK_dX(self,partial,X,X2,target):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        K1 = np.zeros((X.shape[0],X2.shape[0]))
        K2 = np.zeros((X.shape[0],X2.shape[0]))
        self.k1.K(X[:,0:self.k1.D],X2[:,0:self.k1.D],K1)
        self.k2.K(X[:,self.k1.D:],X2[:,self.k1.D:],K2)

        self.k1.dK_dX(partial*K2, X[:,:self.k1.D], X2[:,:self.k1.D], target)
        self.k2.dK_dX(partial*K1, X[:,self.k1.D:], X2[:,self.k1.D:], target)

    def dKdiag_dX(self,X,target):
        pass
