# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np

class Fixed(Kernpart):
    def __init__(self, input_dim, K, variance=1.):
        """
        :param input_dim: the number of input dimensions
        :type input_dim: int
        :param variance: the variance of the kernel
        :type variance: float
        """
        self.input_dim = input_dim
        self.fixed_K = K
        self.num_params = 1
        self.name = 'fixed'
        self._set_params(np.array([variance]).flatten())

    def _get_params(self):
        return self.variance

    def _set_params(self, x):
        assert x.shape == (1,)
        self.variance = x

    def _get_param_names(self):
        return ['variance']

    def K(self, X, X2, target):
        target += self.variance * self.fixed_K

    def _param_grad_helper(self, partial, X, X2, target):
        target += (partial * self.fixed_K).sum()

    def gradients_X(self, partial, X, X2, target):
        pass

    def dKdiag_dX(self, partial, X, target):
        pass
