# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
from ..core.parameterization import Param

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

    def __init__(self, input_dim=1, output_dim=1, name='linear_map'):
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim, name=name)
        self.W = Param('W',np.array((self.input_dim, self.output_dim)))
        self.bias = Param('bias',np.array(self.output_dim))
        self.add_parameters(self.W, self.bias)

    def f(self, X):
        return np.dot(X,self.W) + self.bias

    def df_dtheta(self, dL_df, X):
        df_dW = (dL_df[:, :, None]*X[:, None, :]).sum(0).T
        df_dbias = (dL_df.sum(0))
        return np.hstack((df_dW.flatten(), df_dbias))

    def dL_dX(self, dL_df, X):
        return (dL_df[:, None, :]*self.W[None, :, :]).sum(2)
