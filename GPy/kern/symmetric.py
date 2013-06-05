# Copyright (c) 2012 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np

class symmetric(Kernpart):
    """
    Symmetrical kernels

    :param k: the kernel to symmetrify
    :type k: Kernpart
    :param transform: the transform to use in symmetrification (allows symmetry on specified axes)
    :type transform: A numpy array (input_dim x input_dim) specifiying the transform
    :rtype: Kernpart

    """
    def __init__(self,k,transform=None):
        if transform is None:
            transform = np.eye(k.input_dim)*-1.
        assert transform.shape == (k.input_dim, k.input_dim)
        self.transform = transform
        self.input_dim = k.input_dim
        self.num_params = k.num_params
        self.name = k.name + '_symm'
        self.k = k
        self._set_params(k._get_params())

    def _get_params(self):
        """return the value of the parameters."""
        return self.k._get_params()

    def _set_params(self,x):
        """set the value of the parameters."""
        self.k._set_params(x)

    def _get_param_names(self):
        """return parameter names."""
        return self.k._get_param_names()

    def K(self,X,X2,target):
        """Compute the covariance matrix between X and X2."""
        AX = np.dot(X,self.transform)
        if X2 is None:
            X2 = X
            AX2 = AX
        else:
            AX2 = np.dot(X2, self.transform)
        self.k.K(X,X2,target)
        self.k.K(AX,X2,target)
        self.k.K(X,AX2,target)
        self.k.K(AX,AX2,target)

    def dK_dtheta(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to the parameters."""
        AX = np.dot(X,self.transform)
        if X2 is None:
            X2 = X
            ZX2 = AX
        else:
            AX2 = np.dot(X2, self.transform)
        self.k.dK_dtheta(dL_dK,X,X2,target)
        self.k.dK_dtheta(dL_dK,AX,X2,target)
        self.k.dK_dtheta(dL_dK,X,AX2,target)
        self.k.dK_dtheta(dL_dK,AX,AX2,target)


    def dK_dX(self,dL_dK,X,X2,target):
        """derivative of the covariance matrix with respect to X."""
        AX = np.dot(X,self.transform)
        if X2 is None:
            X2 = X
            ZX2 = AX
        else:
            AX2 = np.dot(X2, self.transform)
        self.k.dK_dX(dL_dK, X, X2, target)
        self.k.dK_dX(dL_dK, AX, X2, target)
        self.k.dK_dX(dL_dK, X, AX2, target)
        self.k.dK_dX(dL_dK, AX ,AX2, target)

    def Kdiag(self,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        foo = np.zeros((X.shape[0],X.shape[0]))
        self.K(X,X,foo)
        target += np.diag(foo)

    def dKdiag_dX(self,dL_dKdiag,X,target):
        raise NotImplementedError

    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        """Compute the diagonal of the covariance matrix associated to X."""
        raise NotImplementedError
