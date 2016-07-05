#===============================================================================
# Copyright (c) 2012-2015, GPy authors (see AUTHORS.txt).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of GPy nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================
import numpy as np
from . import plotting_library as pl
#from .. import gpy_plot
from .plot_util import get_x_y_var, get_free_dims, get_which_data_ycols,\
    get_which_data_rows, update_not_existing_kwargs, helper_predict_with_model

def plot_data(self, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        projection='2d', label=None, **plot_kwargs):
    """
    Plot the training data
      - For higher dimensions than two, use fixed_inputs to plot the data points with some of the inputs fixed.

    Can plot only part of the data
    using which_data_rows and which_data_ycols.

    :param which_data_rows: which of the training data to plot (default all)
    :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
    :param which_data_ycols: when the data has several columns (independant outputs), only plot these
    :type which_data_ycols: 'all' or a list of integers
    :param visible_dims: an array specifying the input dimensions to plot (maximum two)
    :type visible_dims: a numpy array
    :param {'2d','3d'} projection: whether to plot in 2d or 3d. This only applies when plotting two dimensional inputs!
    :param str label: the label for the plot
    :param kwargs plot_kwargs: kwargs for the data plot for the plotting library you are using

    :returns list: of plots created.
    """
    canvas, plot_kwargs = pl().new_canvas(projection=projection, **plot_kwargs)
    plots = _plot_data(self, canvas, which_data_rows, which_data_ycols, visible_dims, projection, label, **plot_kwargs)
    return pl().add_to_canvas(canvas, plots)

def _plot_data(self, canvas, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        projection='2d', label=None, **plot_kwargs):
    ycols = get_which_data_ycols(self, which_data_ycols)
    rows = get_which_data_rows(self, which_data_rows)

    X, _, Y = get_x_y_var(self)
    free_dims = get_free_dims(self, visible_dims, None)

    plots = {}
    plots['dataplot'] = []

    #one dimensional plotting
    if len(free_dims) == 1:
        for d in ycols:
            update_not_existing_kwargs(plot_kwargs, pl().defaults.data_1d)  # @UndefinedVariable
            plots['dataplot'].append(pl().scatter(canvas, X[rows, free_dims], Y[rows, d], label=label, **plot_kwargs))
    #2D plotting
    elif len(free_dims) == 2:
        if projection=='2d':
            for d in ycols:
                update_not_existing_kwargs(plot_kwargs, pl().defaults.data_2d)  # @UndefinedVariable
                plots['dataplot'].append(pl().scatter(canvas, X[rows, free_dims[0]], X[rows, free_dims[1]],
                                               color=Y[rows, d], label=label, **plot_kwargs))
        else:
            for d in ycols:
                update_not_existing_kwargs(plot_kwargs, pl().defaults.data_2d)  # @UndefinedVariable
                plots['dataplot'].append(pl().scatter(canvas, X[rows, free_dims[0]], X[rows, free_dims[1]],
                                                    Z=Y[rows, d], color=Y[rows, d], label=label, **plot_kwargs))
    elif len(free_dims) == 0:
        pass #Nothing to plot!
    else:
        raise NotImplementedError("Cannot plot in more then two dimensions")
    return plots

def plot_data_error(self, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        projection='2d', label=None, **error_kwargs):
    """
    Plot the training data input error.

    For higher dimensions than two, use fixed_inputs to plot the data points with some of the inputs fixed.

    Can plot only part of the data
    using which_data_rows and which_data_ycols.

    :param which_data_rows: which of the training data to plot (default all)
    :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
    :param which_data_ycols: when the data has several columns (independant outputs), only plot these
    :type which_data_ycols: 'all' or a list of integers
    :param visible_dims: an array specifying the input dimensions to plot (maximum two)
    :type visible_dims: a numpy array
    :param {'2d','3d'} projection: whether to plot in 2d or 3d. This only applies when plotting two dimensional inputs!
    :param dict error_kwargs: kwargs for the error plot for the plotting library you are using
    :param str label: the label for the plot
    :param kwargs plot_kwargs: kwargs for the data plot for the plotting library you are using

    :returns list: of plots created.
    """
    canvas, error_kwargs = pl().new_canvas(projection=projection, **error_kwargs)
    plots = _plot_data_error(self, canvas, which_data_rows, which_data_ycols, visible_dims, projection, label, **error_kwargs)
    return pl().add_to_canvas(canvas, plots)

def _plot_data_error(self, canvas, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        projection='2d', label=None, **error_kwargs):
    ycols = get_which_data_ycols(self, which_data_ycols)
    rows = get_which_data_rows(self, which_data_rows)

    X, X_variance, Y = get_x_y_var(self)
    free_dims = get_free_dims(self, visible_dims, None)

    plots = {}

    if X_variance is not None:
        plots['input_error'] = []
        #one dimensional plotting
        if len(free_dims) == 1:
            for d in ycols:
                    update_not_existing_kwargs(error_kwargs, pl().defaults.xerrorbar)
                    plots['input_error'].append(pl().xerrorbar(canvas, X[rows, free_dims].flatten(), Y[rows, d].flatten(),
                                2 * np.sqrt(X_variance[rows, free_dims].flatten()), label=label,
                                **error_kwargs))
        #2D plotting
        elif len(free_dims) == 2:
            update_not_existing_kwargs(error_kwargs, pl().defaults.xerrorbar)  # @UndefinedVariable
            plots['input_error'].append(pl().xerrorbar(canvas, X[rows, free_dims[0]].flatten(), X[rows, free_dims[1]].flatten(),
                            2 * np.sqrt(X_variance[rows, free_dims[0]].flatten()), label=label,
                            **error_kwargs))
            plots['input_error'].append(pl().yerrorbar(canvas, X[rows, free_dims[0]].flatten(), X[rows, free_dims[1]].flatten(),
                            2 * np.sqrt(X_variance[rows, free_dims[1]].flatten()), label=label,
                            **error_kwargs))
        elif len(free_dims) == 0:
            pass #Nothing to plot!
        else:
            raise NotImplementedError("Cannot plot in more then two dimensions")

    return plots

def plot_inducing(self, visible_dims=None, projection='2d', label='inducing', legend=True, **plot_kwargs):
    """
    Plot the inducing inputs of a sparse gp model

    :param array-like visible_dims: an array specifying the input dimensions to plot (maximum two)
    :param kwargs plot_kwargs: keyword arguments for the plotting library
    """
    canvas, kwargs = pl().new_canvas(projection=projection, **plot_kwargs)
    plots = _plot_inducing(self, canvas, visible_dims, projection, label, **kwargs)
    return pl().add_to_canvas(canvas, plots, legend=legend)

def _plot_inducing(self, canvas, visible_dims, projection, label, **plot_kwargs):
    if visible_dims is None:
        sig_dims = self.get_most_significant_input_dimensions()
        visible_dims = [i for i in sig_dims if i is not None]
    free_dims = get_free_dims(self, visible_dims, None)

    Z = self.Z.values
    plots = {}

    #one dimensional plotting
    if len(free_dims) == 1:
        update_not_existing_kwargs(plot_kwargs, pl().defaults.inducing_1d)  # @UndefinedVariable
        plots['inducing'] = pl().plot_axis_lines(canvas, Z[:, free_dims], label=label, **plot_kwargs)
    #2D plotting
    elif len(free_dims) == 2 and projection == '3d':
        update_not_existing_kwargs(plot_kwargs, pl().defaults.inducing_3d)  # @UndefinedVariable
        plots['inducing'] = pl().plot_axis_lines(canvas, Z[:, free_dims], label=label, **plot_kwargs)
    elif len(free_dims) == 2:
        update_not_existing_kwargs(plot_kwargs, pl().defaults.inducing_2d)  # @UndefinedVariable
        plots['inducing'] = pl().scatter(canvas, Z[:, free_dims[0]], Z[:, free_dims[1]],
                                       label=label, **plot_kwargs)
    elif len(free_dims) == 0:
        pass #Nothing to plot!
    else:
        raise NotImplementedError("Cannot plot in more then two dimensions")
    return plots

def plot_errorbars_trainset(self, which_data_rows='all',
        which_data_ycols='all', fixed_inputs=None,
        plot_raw=False, apply_link=False, label=None, projection='2d',
        predict_kw=None, **plot_kwargs):
    """
    Plot the errorbars of the GP likelihood on the training data.
    These are the errorbars after the appropriate
    approximations according to the likelihood are done.

    This also works for heteroscedastic likelihoods.

    Give the Y_metadata in the predict_kw if you need it.

    :param which_data_rows: which of the training data to plot (default all)
    :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
    :param which_data_ycols: when the data has several columns (independant outputs), only plot these
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input dimension i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param dict predict_kwargs: kwargs for the prediction used to predict the right quantiles.
    :param kwargs plot_kwargs: kwargs for the data plot for the plotting library you are using
    """
    canvas, kwargs = pl().new_canvas(projection=projection, **plot_kwargs)
    plots = _plot_errorbars_trainset(self, canvas, which_data_rows, which_data_ycols,
                                     fixed_inputs, plot_raw, apply_link, label, projection, predict_kw, **kwargs)
    return pl().add_to_canvas(canvas, plots)

def _plot_errorbars_trainset(self, canvas,
        which_data_rows='all', which_data_ycols='all',
        fixed_inputs=None,
        plot_raw=False, apply_link=False,
        label=None, projection='2d', predict_kw=None, **plot_kwargs):

    ycols = get_which_data_ycols(self, which_data_ycols)
    rows = get_which_data_rows(self, which_data_rows)

    X, _, Y = get_x_y_var(self)

    if fixed_inputs is None:
        fixed_inputs = []
    free_dims = get_free_dims(self, None, fixed_inputs)

    Xgrid = X.copy()
    for i, v in fixed_inputs:
        Xgrid[:, i] = v

    plots = []

    if len(free_dims)<=2 and projection=='2d':
        update_not_existing_kwargs(plot_kwargs, pl().defaults.yerrorbar)
        if predict_kw is None:
                predict_kw = {}
        if 'Y_metadata' not in predict_kw:
            predict_kw['Y_metadata'] = self.Y_metadata or {}
        mu, percs, _ = helper_predict_with_model(self, Xgrid, plot_raw,
                                          apply_link, (2.5, 97.5),
                                          ycols, predict_kw)
        if len(free_dims)==1:
            for d in ycols:
                plots.append(pl().yerrorbar(canvas, X[rows,free_dims[0]], mu[rows,d],
                                          np.vstack([mu[rows, d] - percs[0][rows, d], percs[1][rows, d] - mu[rows,d]]),
                                          label=label,
                                          **plot_kwargs))
#         elif len(free_dims) == 2:
#             for d in ycols:
#                 plots.append(pl().yerrorbar(canvas, X[rows,free_dims[0]], X[rows,free_dims[1]],
#                               np.vstack([mu[rows, d] - percs[0][rows, d], percs[1][rows, d] - mu[rows,d]]),
#                               #color=Y[rows,d],
#                               label=label,
#                               **plot_kwargs))
#                 plots.append(pl().xerrorbar(canvas, X[rows,free_dims[0]], X[rows,free_dims[1]],
#                               np.vstack([mu[rows, d] - percs[0][rows, d], percs[1][rows, d] - mu[rows,d]]),
#                               #color=Y[rows,d],
#                               label=label,
#                               **plot_kwargs))
    else:
        raise NotImplementedError("Cannot plot in more then one dimensions, or 3d")
    return dict(yerrorbars=plots)


