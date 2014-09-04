# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
import GPy

class Additive(Mapping):
    """
    Mapping based on adding two existing mappings together.

    .. math::

       f(\mathbf{x}*) = f_1(\mathbf{x}*) + f_2(\mathbf(x)*)

    :param mapping1: first mapping to add together.
    :type mapping1: GPy.mappings.Mapping
    :param mapping2: second mapping to add together.
    :type mapping2: GPy.mappings.Mapping
    :param tensor: whether or not to use the tensor product of input spaces
    :type tensor: bool

    """

    def __init__(self, mapping1, mapping2, tensor=False):
        if tensor:
            input_dim = mapping1.input_dim + mapping2.input_dim
        else:
            input_dim = mapping1.input_dim
            assert(mapping1.input_dim==mapping2.input_dim)
        assert(mapping1.output_dim==mapping2.output_dim)
        output_dim = mapping1.output_dim
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim)
        self.mapping1 = mapping1
        self.mapping2 = mapping2
        self.num_params = self.mapping1.num_params + self.mapping2.num_params
        self.name = self.mapping1.name + '+' + self.mapping2.name
    def _get_param_names(self):
        return self.mapping1._get_param_names + self.mapping2._get_param_names

    def _get_params(self):
        return np.hstack((self.mapping1._get_params(), self.mapping2._get_params()))

    def _set_params(self, x):
        self.mapping1._set_params(x[:self.mapping1.num_params])
        self.mapping2._set_params(x[self.mapping1.num_params:])
        
    def randomize(self):
        self.mapping1._randomize()
        self.mapping2._randomize()

    def f(self, X):
        return self.mapping1.f(X) + self.mapping2.f(X)

    def df_dtheta(self, dL_df, X):
        self._df_dA = (dL_df[:, :, None]*self.kern.K(X, self.X)[:, None, :]).sum(0).T
        self._df_dbias = (dL_df.sum(0))
        return np.hstack((self._df_dA.flatten(), self._df_dbias))

    def df_dX(self, dL_df, X):
        return self.kern.dK_dX((dL_df[:, None, :]*self.A[None, :, :]).sum(2), X, self.X) 
