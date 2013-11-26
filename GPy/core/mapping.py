# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..util.plot import Tango, x_frame1D, x_frame2D
from parameterized import Parameterized
import numpy as np
import pylab as pb

class Mapping(Parameterized):
    """
    Base model for shared behavior between models that can act like a mapping. 
    """

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        super(Mapping, self).__init__()
        # Model.__init__(self)
        # All leaf nodes should call self._set_params(self._get_params()) at
        # the end

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
        """The gradient of the outputs of the multi-layer perceptron with respect to each of the parameters.
        :param dL_df: gradient of the objective with respect to the function.
        :type dL_df: ndarray (num_data x output_dim)
        :param X: input locations where the function is evaluated.
        :type X: ndarray (num_data x input_dim)
        :returns: Matrix containing gradients with respect to parameters of each output for each input data.
        :rtype: ndarray (num_params length)
        """
        raise NotImplementedError

    def plot(self, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20, samples=0, fignum=None, ax=None, fixed_inputs=[], linecol=Tango.colorsHex['darkBlue']):
        """

        Plot the mapping.

        Plots the mapping associated with the model.
          - In one dimension, the function is plotted.
          - In two dimsensions, a contour-plot shows the function
          - In higher dimensions, we've not implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions
        using which_data and which_functions

        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :type plot_limits: np.array
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param levels: number of levels to plot in a contour plot.
        :type levels: int
        :param samples: the number of a posteriori samples to plot
        :type samples: int
        :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        :type fixed_inputs: a list of tuples
        :param linecol: color of line to plot.
        :type linecol:
        :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure

        """
        # TODO include samples
        if which_data == 'all':
            which_data = slice(None)

        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        plotdims = self.input_dim - len(fixed_inputs)

        if plotdims == 1:

            Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now

            fixed_dims = np.array([i for i,v in fixed_inputs])
            freedim = np.setdiff1d(np.arange(self.input_dim),fixed_dims)

            Xnew, xmin, xmax = x_frame1D(Xu[:,freedim], plot_limits=plot_limits)
            Xgrid = np.empty((Xnew.shape[0],self.input_dim))
            Xgrid[:,freedim] = Xnew
            for i,v in fixed_inputs:
                Xgrid[:,i] = v

            f = self.predict(Xgrid, which_parts=which_parts)
            for d in range(y.shape[1]):
                ax.plot(Xnew, f[:, d], edgecol=linecol)

        elif self.X.shape[1] == 2:
            resolution = resolution or 50
            Xnew, _, _, xmin, xmax = x_frame2D(self.X, plot_limits, resolution)
            x, y = np.linspace(xmin[0], xmax[0], resolution), np.linspace(xmin[1], xmax[1], resolution)
            f = self.predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            ax.contour(x, y, f, levels, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet) # @UndefinedVariable
            ax.set_xlim(xmin[0], xmax[0])
            ax.set_ylim(xmin[1], xmax[1])

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

from GPy.core.model import Model

class Mapping_check_model(Model):
    """
    This is a dummy model class used as a base class for checking that the
    gradients of a given mapping are implemented correctly. It enables
    checkgradient() to be called independently on each mapping.
    """
    def __init__(self, mapping=None, dL_df=None, X=None):
        num_samples = 20
        if mapping==None:
            mapping = GPy.mapping.linear(1, 1)
        if X==None:
            X = np.random.randn(num_samples, mapping.input_dim)
        if dL_df==None:
            dL_df = np.ones((num_samples, mapping.output_dim))

        self.mapping=mapping
        self.X = X
        self.dL_df = dL_df
        self.num_params = self.mapping.num_params
        Model.__init__(self)


    def _get_params(self):
        return self.mapping._get_params()

    def _get_param_names(self):
        return self.mapping._get_param_names()

    def _set_params(self, x):
        self.mapping._set_params(x)

    def log_likelihood(self):
        return (self.dL_df*self.mapping.f(self.X)).sum()

    def _log_likelihood_gradients(self):
        raise NotImplementedError, "This needs to be implemented to use the Mapping_check_model class."

class Mapping_check_df_dtheta(Mapping_check_model):
    """This class allows gradient checks for the gradient of a mapping with respect to parameters. """
    def __init__(self, mapping=None, dL_df=None, X=None):
        Mapping_check_model.__init__(self,mapping=mapping,dL_df=dL_df, X=X)

    def _log_likelihood_gradients(self):
        return self.mapping.df_dtheta(self.dL_df, self.X)


class Mapping_check_df_dX(Mapping_check_model):
    """This class allows gradient checks for the gradient of a mapping with respect to X. """
    def __init__(self, mapping=None, dL_df=None, X=None):
        Mapping_check_model.__init__(self,mapping=mapping,dL_df=dL_df, X=X)

        if dL_df==None:
            dL_df = np.ones((self.X.shape[0],self.mapping.output_dim))
        self.num_params = self.X.shape[0]*self.mapping.input_dim

    def _log_likelihood_gradients(self):
        return self.mapping.df_dX(self.dL_df, self.X).flatten()

    def _get_param_names(self):
        return ['X_'  +str(i) + ','+str(j) for j in range(self.X.shape[1]) for i in range(self.X.shape[0])]

    def _get_params(self):
        return self.X.flatten()

    def _set_params(self, x):
        self.X=x.reshape(self.X.shape)

