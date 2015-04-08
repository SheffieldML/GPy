# Copyright (c) 2015, James Hensman and Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core import Mapping

class Compound(Mapping):
    """
    Mapping based on passing one mapping through another

    .. math::

       f(\mathbf{x}) = f_2(f_1(\mathbf{x}))

    :param mapping1: first mapping
    :type mapping1: GPy.mappings.Mapping
    :param mapping2: second mapping
    :type mapping2: GPy.mappings.Mapping

    """

    def __init__(self, mapping1, mapping2):
        assert(mapping1.output_dim==mapping2.input_dim)
        input_dim, output_dim = mapping1.input_dim, mapping2.output_dim
        Mapping.__init__(self, input_dim=input_dim, output_dim=output_dim)
        self.mapping1 = mapping1
        self.mapping2 = mapping2
        self.link_parameters(self.mapping1, self.mapping2)

    def f(self, X):
        return self.mapping2.f(self.mapping1.f(X))

    def update_gradients(self, dL_dF, X):
        hidden = self.mapping1.f(X)
        self.mapping2.update_gradients(dL_dF, hidden)
        self.mapping1.update_gradients(self.mapping2.gradients_X(dL_dF, hidden), X)

    def gradients_X(self, dL_dF, X):
        hidden = self.mapping1.f(X)
        return self.mapping1.gradients_X(self.mapping2.gradients_X(dL_dF, hidden), X)
