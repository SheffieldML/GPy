# Copyright (c) 2015, James Hensman

from ..core.mapping import Mapping
from ..core import Param

class Identity(Mapping):
    """
    A mapping that does nothing!
    """
    def __init__(self, input_dim, output_dim, name='identity'):
        Mapping.__init__(self, input_dim, output_dim, name)

    def f(self, X):
        return X

    def update_gradients(self, dL_dF, X):
        pass

    def gradients_X(self, dL_dF, X):
        return dL_dF

    def to_dict(self):
        input_dict = super(Identity, self)._to_dict()
        input_dict["class"] = "GPy.mappings.Identity"
        return input_dict
