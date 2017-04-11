# Copyright (c) 2013,2014, GPy authors (see AUTHORS.txt).
# Copyright (c) 2015, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
from .parameterization import Parameterized
import numpy as np

class Mapping(Parameterized):
    """
    Base model for shared mapping behaviours
    """

    def __init__(self, input_dim, output_dim, name='mapping'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Mapping, self).__init__(name=name)

    def f(self, X):
        raise NotImplementedError

    def gradients_X(self, dL_dF, X):
        raise NotImplementedError

    def update_gradients(self, dL_dF, X):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def _to_dict(self):
        input_dict = {}
        input_dict["input_dim"] = self.input_dim
        input_dict["output_dim"] = self.output_dim
        input_dict["name"] = self.name
        return input_dict

    @staticmethod
    def from_dict(input_dict):
        import copy
        input_dict = copy.deepcopy(input_dict)
        mapping_class = input_dict.pop('class')
        input_dict["name"] = str(input_dict["name"])
        import GPy
        mapping_class = eval(mapping_class)
        return mapping_class._from_dict(mapping_class, input_dict)

    @staticmethod
    def _from_dict(mapping_class, input_dict):
        return mapping_class(**input_dict)


class Bijective_mapping(Mapping):
    """
    This is a mapping that is bijective, i.e. you can go from X to f and
    also back from f to X. The inverse mapping is called g().
    """
    def __init__(self, input_dim, output_dim, name='bijective_mapping'):
        super(Bijective_mapping, self).__init__(name=name)

    def g(self, f):
        """Inverse mapping from output domain of the function to the inputs."""
        raise NotImplementedError
