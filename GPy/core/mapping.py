# Copyright (c) 2013,2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
from parameterization import Parameterized
import numpy as np

class Mapping(Parameterized):
    """
    Base model for shared behavior between models that can act like a mapping.
    """

    def __init__(self, input_dim, output_dim, name='mapping'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Mapping, self).__init__(name=name)

    def f(self, X):
        raise NotImplementedError

    def df_dX(self, dL_df, X):
        """Evaluate derivatives of mapping outputs with respect to inputs.

        :param dL_df: gradient of the objective with respect to the function.
        :type dL_df: ndarray (num_data x output_dim)
        :param X: the input locations where derivatives are to be evaluated.
        :type X: ndarray (num_data x input_dim)
        :returns: matrix containing gradients of the function with respect to the inputs.
        """
        raise NotImplementedError

    def df_dtheta(self, dL_df, X):
        """The gradient of the outputs of the mapping with respect to each of the parameters.

        :param dL_df: gradient of the objective with respect to the function.
        :type dL_df: ndarray (num_data x output_dim)
        :param X: input locations where the function is evaluated.
        :type X: ndarray (num_data x input_dim)
        :returns: Matrix containing gradients with respect to parameters of each output for each input data.
        :rtype: ndarray (num_params length)
        """

        raise NotImplementedError

    def plot(self, *args):
        """
        Plots the mapping associated with the model.
          - In one dimension, the function is plotted.
          - In two dimensions, a contour-plot shows the function
          - In higher dimensions, we've not implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions
        using which_data and which_functions

        This is a convenience function: arguments are passed to
        GPy.plotting.matplot_dep.models_plots.plot_mapping
        """

        if "matplotlib" in sys.modules:
            from ..plotting.matplot_dep import models_plots
            mapping_plots.plot_mapping(self,*args)
        else:
            raise NameError, "matplotlib package has not been imported."

class Bijective_mapping(Mapping):
    """
    This is a mapping that is bijective, i.e. you can go from X to f and
    also back from f to X. The inverse mapping is called g().
    """
    def __init__(self, input_dim, output_dim, name='bijective_mapping'):
        super(Bijective_apping, self).__init__(name=name)

    def g(self, f):
        """Inverse mapping from output domain of the function to the inputs."""
        raise NotImplementedError


