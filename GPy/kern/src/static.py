# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .kern import Kern
import numpy as np
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp

class Static(Kern):
    def __init__(self, input_dim, variance, active_dims, name):
        super(Static, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', variance, Logexp())
        self.link_parameters(self.variance)

    def Kdiag(self, X):
        ret = np.empty((X.shape[0],), dtype=np.float64)
        ret[:] = self.variance
        return ret

    def gradients_X(self, dL_dK, X, X2=None):
        return np.zeros(X.shape)

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)

    def gradients_XX(self, dL_dK, X, X2):
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0], X2.shape[0], X.shape[1]), dtype=np.float64)
    def gradients_XX_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)

    def gradients_Z_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return np.zeros(Z.shape)

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        return np.zeros(variational_posterior.shape), np.zeros(variational_posterior.shape)

    def psi0(self, Z, variational_posterior):
        return self.Kdiag(variational_posterior.mean)

    def psi1(self, Z, variational_posterior):
        return self.K(variational_posterior.mean, Z)

    def psi2(self, Z, variational_posterior):
        K = self.K(variational_posterior.mean, Z)
        return np.einsum('ij,ik->jk',K,K) #K[:,:,None]*K[:,None,:] # NB. more efficient implementations on inherriting classes

    def input_sensitivity(self, summarize=True):
        if summarize:
            return super(Static, self).input_sensitivity(summarize=summarize)
        else:
            return np.ones(self.input_dim) * self.variance

class White(Static):
    def __init__(self, input_dim, variance=1., active_dims=None, name='white'):
        super(White, self).__init__(input_dim, variance, active_dims, name)

    def K(self, X, X2=None):
        if X2 is None:
            return np.eye(X.shape[0])*self.variance
        else:
            return np.zeros((X.shape[0], X2.shape[0]))

    def psi2(self, Z, variational_posterior):
        return np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float64)

    def psi2n(self, Z, variational_posterior):
        return np.zeros((1, Z.shape[0], Z.shape[0]), dtype=np.float64)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            self.variance.gradient = np.trace(dL_dK)
        else:
            self.variance.gradient = 0.

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = dL_dKdiag.sum()

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        self.variance.gradient = dL_dpsi0.sum()

class Bias(Static):
    def __init__(self, input_dim, variance=1., active_dims=None, name='bias'):
        super(Bias, self).__init__(input_dim, variance, active_dims, name)

    def K(self, X, X2=None):
        shape = (X.shape[0], X.shape[0] if X2 is None else X2.shape[0])
        ret = np.empty(shape, dtype=np.float64)
        ret[:] = self.variance
        return ret

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = dL_dK.sum()

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = dL_dKdiag.sum()

    def psi2(self, Z, variational_posterior):
        ret = np.empty((Z.shape[0], Z.shape[0]), dtype=np.float64)
        ret[:] = self.variance*self.variance*variational_posterior.shape[0]
        return ret

    def psi2n(self, Z, variational_posterior):
        ret = np.empty((variational_posterior.mean.shape[0], Z.shape[0], Z.shape[0]), dtype=np.float64)
        ret[:] = self.variance*self.variance
        return ret

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        if dL_dpsi2.ndim == 2:
            self.variance.gradient = (dL_dpsi0.sum() + dL_dpsi1.sum()
                                    + 2.*self.variance*dL_dpsi2.sum()*variational_posterior.shape[0])
        else:
            self.variance.gradient = (dL_dpsi0.sum() + dL_dpsi1.sum()
                                    + 2.*self.variance*dL_dpsi2.sum())

class Fixed(Static):
    def __init__(self, input_dim, covariance_matrix, variance=1., active_dims=None, name='fixed'):
        """
        :param input_dim: the number of input dimensions
        :type input_dim: int
        :param variance: the variance of the kernel
        :type variance: float
        """
        super(Fixed, self).__init__(input_dim, variance, active_dims, name)
        self.fixed_K = covariance_matrix
    def K(self, X, X2):
        return self.variance * self.fixed_K

    def Kdiag(self, X):
        return self.variance * self.fixed_K.diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.einsum('ij,ij', dL_dK, self.fixed_K)

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.einsum('i,i', dL_dKdiag, self.fixed_K)

    def psi2(self, Z, variational_posterior):
        return np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float64)

    def psi2n(self, Z, variational_posterior):
        return np.zeros((1, Z.shape[0], Z.shape[0]), dtype=np.float64)

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        self.variance.gradient = dL_dpsi0.sum()

