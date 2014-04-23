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
    def __init__(self, input_dim, output_dim, f=None, name='symbolic', parameters=None, func_modules=[]):


        if f is None:
            raise ValueError, "You must provide an argument for the function."

        Mapping.__init__(self, input_dim, output_dim, name=name)
        Symbolic_core.__init__(self, {'f': f}, ['X'], derivatives = ['X', 'theta'], parameters=parameters, func_modules=func_modules)

        self._initialize_cache()
        self.parameters_changed()

    def _initialize_cache(self):
        self._set_attribute('x_0', np.random.normal(size=(3, self.input_dim)))
        

    def parameters_changed(self):
        self.eval_parameters_changed()

    def update_cache(self, X=None):
        self.eval_update_cache(X=X)

    def update_gradients(self, partial, X=None):
        for name, val in self.eval_update_gradients('f', partial, X=X).iteritems():
            setattr(getattr(self, name), 'gradient', val)

    def gradients_X(self, partial, X=None):
        return self.eval_gradients_X('f', partial, X=X)

    def f(self, X=None):
        """
        """
        return self.eval_function('f', X=X)


    def df_dX(self, X):
        """
        """
        pass

    def df_dtheta(self, X):
        pass
