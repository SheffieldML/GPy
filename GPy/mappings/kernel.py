# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
import GPy

class Kernel(Mapping):
    """
    Mapping based on a kernel/covariance function.

    .. math::

       f(\mathbf{x}*) = \mathbf{A}\mathbf{k}(\mathbf{X}, \mathbf{x}^*) + \mathbf{b}

    :param X: input observations containing :math:`\mathbf{X}`
    :type X: ndarray
    :param output_dim: dimension of output.
    :type output_dim: int
    :param kernel: a GPy kernel, defaults to GPy.kern.RBF
    :type kernel: GPy.kern.kern

    """

    def __init__(self, X, output_dim=1, kernel=None):
        Mapping.__init__(self, input_dim=X.shape[1], output_dim=output_dim)
        if kernel is None:
            kernel = GPy.kern.RBF(self.input_dim)
        self.kern = kernel
        self.X = X
        self.num_data = X.shape[0]
        self.num_params = self.output_dim*(self.num_data + 1)
        self.A = np.array((self.num_data, self.output_dim))
        self.bias = np.array(self.output_dim)
        self.randomize()
        self.name = 'kernel'
    def _get_param_names(self):
        return sum([['A_%i_%i' % (n, d) for d in range(self.output_dim)] for n in range(self.num_data)], []) + ['bias_%i' % d for d in range(self.output_dim)]

    def _get_params(self):
        return np.hstack((self.A.flatten(), self.bias))

    def _set_params(self, x):
        self.A = x[:self.num_data * self.output_dim].reshape(self.num_data, self.output_dim).copy()
        self.bias = x[self.num_data*self.output_dim:].copy()

    def randomize(self):
        self.A = np.random.randn(self.num_data, self.output_dim)/np.sqrt(self.num_data+1)
        self.bias = np.random.randn(self.output_dim)/np.sqrt(self.num_data+1)

    def f(self, X):
        return np.dot(self.kern.K(X, self.X),self.A) + self.bias

    def df_dtheta(self, dL_df, X):
        self._df_dA = (dL_df[:, :, None]*self.kern.K(X, self.X)[:, None, :]).sum(0).T
        self._df_dbias = (dL_df.sum(0))
        return np.hstack((self._df_dA.flatten(), self._df_dbias))

    def df_dX(self, dL_df, X):
        return self.kern.gradients_X((dL_df[:, None, :]*self.A[None, :, :]).sum(2), X, self.X)
