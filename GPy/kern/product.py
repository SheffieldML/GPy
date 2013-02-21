# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import kernpart
import numpy as np
import hashlib
#from scipy import integrate # This may not be necessary (Nicolas, 20th Feb)

class product(kernpart):
    """
    Computes the product of 2 kernels that are defined on the same space

    :param k1, k2: the kernels to multiply
    :type k1, k2: kernpart
    :rtype: kernel object

    """
    def __init__(self,k1,k2):
        assert k1.D == k2.D, "Error: The input spaces of the kernels to multiply must have the same dimension"
        self.D = k1.D
        self.Nparam = k1.Nparam + k2.Nparam
        self.name = k1.name + '<times>' + k2.name
        self.k1 = k1
        self.k2 = k2
        self._set_params(np.hstack((k1._get_params(),k2._get_params())))

    def _get_params(self):
        """return the value of the parameters."""
        return self.params

    def _set_params(self,x):
        """set the value of the parameters."""
        self.k1._set_params(x[:self.k1.Nparam])
        self.k2._set_params(x[self.k1.Nparam:])
        self.params = x

    def _get_param_names(self):
        """return parameter names."""
        return [self.k1.name + '_' + param_name for param_name in self.k1._get_param_names()] + [self.k2.name + '_' + param_name for param_name in self.k2._get_param_names()]

    def K(self,X,X2,target):
        """Compute the covariance matrix between X and X2."""
        if X2 is None: X2 = X
        target1 = np.zeros((X.shape[0],X2.shape[0]))
        target2 = np.zeros((X.shape[0],X2.shape[0]))
        self.k1.K(X,X2,target1)
        self.k2.K(X,X2,target2)
        target += target1 * target2

    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        target1 = np.zeros((X.shape[0],))
        target2 = np.zeros((X.shape[0],))
        self.k1.Kdiag(X,target1)
        self.k2.Kdiag(X,target2)
        target += target1 * target2

    def dK_dtheta(self,partial,X,X2,target):
        """derivative of the covariance matrix with respect to the parameters."""
        if X2 is None: X2 = X
        K1 = np.zeros((X.shape[0],X2.shape[0]))
        K2 = np.zeros((X.shape[0],X2.shape[0]))
        self.k1.K(X,X2,K1)
        self.k2.K(X,X2,K2)

        k1_target = np.zeros(self.k1.Nparam)
        k2_target = np.zeros(self.k2.Nparam)
        self.k1.dK_dtheta(partial*K2, X, X2, k1_target)
        self.k2.dK_dtheta(partial*K1, X, X2, k2_target)

        target[:self.k1.Nparam] += k1_target
        target[self.k1.Nparam:] += k2_target

    def dK_dX(self,partial,X,X2,target):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        K1 = np.zeros((X.shape[0],X2.shape[0]))
        K2 = np.zeros((X.shape[0],X2.shape[0]))
        self.k1.K(X,X2,K1)
        self.k2.K(X,X2,K2)

        self.k1.dK_dX(partial*K2, X, X2, target)
        self.k2.dK_dX(partial*K1, X, X2, target)

    def dKdiag_dX(self,X,target):
        pass
