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
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        input_dict = super(Identity, self)._save_to_input_dict()
        input_dict["class"] = "GPy.mappings.Identity"
        return input_dict
