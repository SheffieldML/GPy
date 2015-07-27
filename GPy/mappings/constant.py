# Copyright (c) 2015, James Hensman, Alan Saul
import numpy as np
from ..core.mapping import Mapping
from ..core.parameterization import Param

class Constant(Mapping):
    """
    A Linear mapping.

    .. math::

       F(\mathbf{x}) = c


    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    :param: value the value of this constant mapping

    """

    def __init__(self, input_dim, output_dim, value=0., name='constmap'):
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim, name=name)
        value = np.atleast_1d(value)
        if not len(value.shape) ==1:
            raise ValueError("bad constant values: pass a float or flat vectoor")
        elif value.size==1:
            value = np.ones(self.output_dim)*value
        self.C = Param('C', value)
        self.link_parameter(self.C)

    def f(self, X):
        return np.tile(self.C.values[None,:], (X.shape[0], 1))

    def update_gradients(self, dL_dF, X):
        self.C.gradient = dL_dF.sum(0)

    def gradients_X(self, dL_dF, X):
        return np.zeros_like(X)
