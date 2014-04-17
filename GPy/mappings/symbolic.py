# Copyright (c) 2014 GPy Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sympy as sym
from ..core.mapping import Mapping, Bijective_mapping
from ..core.symbolic import Symbolic_core
import numpy as np

class Symbolic(Mapping, Symbolic_core):
    """
    Symbolic mapping

    Mapping where the form of the mapping is provided by a sympy expression.

    """
    def __init__(self, input_dim, output_dim, f=None, name='symbolic', param=None, func_modules=[]):


        if f is None:
            raise ValueError, "You must provide an argument for the function."

        Mapping.__init__(self, input_dim, output_dim, name=name)
        Symbolic_core.__init__(self, f, ['X'], derivatives = ['X', 'theta'], param=param, func_modules=func_modules)

        self._initialize_cache()
        self.parameters_changed()

    def _initialize_cache(self):
        self.x_0 = np.random.normal(size=(3, self.input_dim))


    def parameters_changed(self):
        self.eval_parameters_changed()

    def update_cache(self, X):
        self.eval_update_cache(X)

    def update_gradients(self, partial, X):
        self.eval_update_gradients(partial, X)

    def gradients_X(self, partial, X):
        return self.eval_gradients_X(partial, X)

    def f(self, X):
        """
        """
        return self.eval_f(X)


    def df_dX(self, X):
        """
        """
        pass

    def df_dtheta(self, X):
        pass
