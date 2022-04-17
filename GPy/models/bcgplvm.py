# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from . import GPLVM
from .. import mappings


class BCGPLVM(GPLVM):
    """
    Back constrained Gaussian Process Latent Variable Model

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param mapping: mapping for back constraint
    :type mapping: GPy.core.Mapping object

    """
    def __init__(self, Y, input_dim, kernel=None, mapping=None):


        if mapping is None:
            mapping = mappings.MLP(input_dim=Y.shape[1],
                                   output_dim=input_dim,
                                   hidden_dim=10)
        else:
            assert mapping.input_dim==Y.shape[1], "mapping input dim does not work for Y dimension"
            assert mapping.output_dim==input_dim, "mapping output dim does not work for self.input_dim"
        super(BCGPLVM, self).__init__(Y, input_dim, X=mapping.f(Y), kernel=kernel, name="bcgplvm")
        self.unlink_parameter(self.X)
        self.mapping = mapping
        self.link_parameter(self.mapping)

        self.X = self.mapping.f(self.Y)

    def parameters_changed(self):
        self.X = self.mapping.f(self.Y)
        GP.parameters_changed(self)
        Xgradient = self.kern.gradients_X(self.grad_dict['dL_dK'], self.X, None)
        self.mapping.update_gradients(Xgradient, self.Y)


