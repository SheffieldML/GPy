# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping

class Linear(Mapping):
    """
    Mapping based on a linear model.

    .. math::

       f(\mathbf{x}*) = \mathbf{W}\mathbf{x}^* + \mathbf{b}

    :param X: input observations
    :type X: ndarray
    :param output_dim: dimension of output.
    :type output_dim: int
    
    """

    def __init__(self, input_dim=1, output_dim=1):
        self.name = 'linear'
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim)
        self.num_params = self.output_dim*(self.input_dim + 1)
        self.W = np.array((self.input_dim, self.output_dim))
        self.bias = np.array(self.output_dim)
        self.randomize()

    def _get_param_names(self):
        return sum([['W_%i_%i' % (n, d) for d in range(self.output_dim)] for n in range(self.input_dim)], []) + ['bias_%i' % d for d in range(self.output_dim)]

    def _get_params(self):
        return np.hstack((self.W.flatten(), self.bias))

    def _set_params(self, x):
        self.W = x[:self.input_dim * self.output_dim].reshape(self.input_dim, self.output_dim).copy()
        self.bias = x[self.input_dim*self.output_dim:].copy()
    def randomize(self):
        self.W = np.random.randn(self.input_dim, self.output_dim)/np.sqrt(self.input_dim + 1)
        self.bias = np.random.randn(self.output_dim)/np.sqrt(self.input_dim + 1)

    def f(self, X):
        return np.dot(X,self.W) + self.bias

    def df_dtheta(self, dL_df, X):
        self._df_dW = (dL_df[:, :, None]*X[:, None, :]).sum(0).T
        self._df_dbias = (dL_df.sum(0))
        return np.hstack((self._df_dW.flatten(), self._df_dbias))
        
    def df_dX(self, dL_df, X):
        return (dL_df[:, None, :]*self.W[None, :, :]).sum(2) 
    
