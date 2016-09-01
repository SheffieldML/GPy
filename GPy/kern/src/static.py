# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .kern import Kern
import numpy as np
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

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

    def gradients_XX(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0], X2.shape[0], X.shape[1], X.shape[1]), dtype=np.float64)

    def gradients_XX_diag(self, dL_dKdiag, X, cov=False):
        return np.zeros((X.shape[0], X.shape[1], X.shape[1]), dtype=np.float64)

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

class WhiteHeteroscedastic(Static):
    def __init__(self, input_dim, num_data, variance=1., active_dims=None, name='white_hetero'):
        """
        A heteroscedastic White kernel (nugget/noise).
        It defines one variance (nugget) per input sample.

        Prediction excludes any noise learnt by this Kernel, so be careful using this kernel.

        You can plot the errors learnt by this kernel by something similar as:
        plt.errorbar(m.X, m.Y, yerr=2*np.sqrt(m.kern.white.variance))
        """
        super(Static, self).__init__(input_dim, active_dims, name)
        self.variance = Param('variance', np.ones(num_data) * variance, Logexp())
        self.link_parameters(self.variance)

    def Kdiag(self, X):
        if X.shape[0] == self.variance.shape[0]:
            # If the input has the same number of samples as
            # the number of variances, we return the variances
            return self.variance
        return 0.

    def K(self, X, X2=None):
        if X2 is None and X.shape[0] == self.variance.shape[0]:
            return np.eye(X.shape[0]) * self.variance
        else:
            return 0.

    def psi2(self, Z, variational_posterior):
        return np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float64)

    def psi2n(self, Z, variational_posterior):
        return np.zeros((1, Z.shape[0], Z.shape[0]), dtype=np.float64)

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            self.variance.gradient = np.diagonal(dL_dK)
        else:
            self.variance.gradient = 0.

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = dL_dKdiag

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        self.variance.gradient = dL_dpsi0

class Bias(Static):
    def __init__(self, input_dim, variance=1., active_dims=None, name='bias'):
        super(Bias, self).__init__(input_dim, variance, active_dims, name)

    def K(self, X, X2=None):
        shape = (X.shape[0], X.shape[0] if X2 is None else X2.shape[0])
        return np.full(shape, self.variance, dtype=np.float64)

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = dL_dK.sum()

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = dL_dKdiag.sum()

    def psi2(self, Z, variational_posterior):
        return np.full((Z.shape[0], Z.shape[0]), self.variance*self.variance*variational_posterior.shape[0], dtype=np.float64)

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
        if X2 is None:
            return self.variance * self.fixed_K
        else:
            return np.zeros((X.shape[0], X2.shape[0]))

    def Kdiag(self, X):
        return self.variance * self.fixed_K.diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            self.variance.gradient = np.einsum('ij,ij', dL_dK, self.fixed_K)
        else:
            self.variance.gradient = 0

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.einsum('i,i', dL_dKdiag, np.diagonal(self.fixed_K))

    def psi2(self, Z, variational_posterior):
        return np.zeros((Z.shape[0], Z.shape[0]), dtype=np.float64)

    def psi2n(self, Z, variational_posterior):
        return np.zeros((1, Z.shape[0], Z.shape[0]), dtype=np.float64)

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        self.variance.gradient = dL_dpsi0.sum()

class Precomputed(Fixed):
    def __init__(self, input_dim, covariance_matrix, variance=1., active_dims=None, name='precomputed'):
        """
        Class for precomputed kernels, indexed by columns in X

        Usage example:

        import numpy as np
        from GPy.models import GPClassification
        from GPy.kern import Precomputed
        from sklearn.cross_validation import LeaveOneOut

        n = 10
        d = 100
        X = np.arange(n).reshape((n,1))         # column vector of indices
        y = 2*np.random.binomial(1,0.5,(n,1))-1
        X0 = np.random.randn(n,d)
        k = np.dot(X0,X0.T)
        kern = Precomputed(1,k)                 # k is a n x n covariance matrix

        cv = LeaveOneOut(n)
        ypred = y.copy()
        for train, test in cv:
            m = GPClassification(X[train], y[train], kernel=kern)
            m.optimize()
            ypred[test] = 2*(m.predict(X[test])[0]>0.5)-1

        :param input_dim: the number of input dimensions
        :type input_dim: int
        :param variance: the variance of the kernel
        :type variance: float
        """
        assert input_dim==1, "Precomputed only implemented in one dimension. Use multiple Precomputed kernels to have more dimensions by making use of active_dims"
        super(Precomputed, self).__init__(input_dim, covariance_matrix, variance, active_dims, name)

    @Cache_this(limit=2)
    def _index(self, X, X2):
        if X2 is None:
            i1 = i2 = X.astype('int').flat
        else:
            i1, i2 = X.astype('int').flat, X2.astype('int').flat
        return self.fixed_K[i1,:][:,i2]

    def K(self, X, X2=None):
        return self.variance * self._index(X, X2)

    def Kdiag(self, X):
        return self.variance * self._index(X,None).diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        self.variance.gradient = np.einsum('ij,ij', dL_dK, self._index(X, X2))

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.einsum('i,ii', dL_dKdiag, self._index(X, None))

