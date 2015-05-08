# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.mapping import Mapping
from ..core import Param

class Kernel(Mapping):
    """
    Mapping based on a kernel/covariance function.

    .. math::

       f(\mathbf{x}) = \sum_i \alpha_i k(\mathbf{z}_i, \mathbf{x})

    or for multple outputs

    .. math::

       f_i(\mathbf{x}) = \sum_j \alpha_{i,j} k(\mathbf{z}_i, \mathbf{x})


    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    :param Z: input observations containing :math:`\mathbf{Z}`
    :type Z: ndarray
    :param kernel: a GPy kernel, defaults to GPy.kern.RBF
    :type kernel: GPy.kern.kern

    """

    def __init__(self, input_dim, output_dim, Z, kernel, name='kernmap'):
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim, name=name)
        self.kern = kernel
        self.Z = Z
        self.num_bases, Zdim = Z.shape
        assert Zdim == self.input_dim
        self.A = Param('A', np.random.randn(self.num_bases, self.output_dim))
        self.link_parameter(self.A)

    def f(self, X):
        return np.dot(self.kern.K(X, self.Z), self.A)

    def update_gradients(self, dL_dF, X):
        self.kern.update_gradients_full(np.dot(dL_dF, self.A.T), X, self.Z)
        self.A.gradient = np.dot( self.kern.K(self.Z, X), dL_dF)

    def gradients_X(self, dL_dF, X):
        return self.kern.gradients_X(np.dot(dL_dF, self.A.T), X, self.Z)
