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

from . import pl

import numpy as np
from .plot_util import get_x_y_var, get_free_dims, get_which_data_ycols,\
    get_which_data_rows, update_not_existing_kwargs

def _plot_data(self, canvas, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        error_kwargs=None, **plot_kwargs):
    if error_kwargs is None:
        error_kwargs = {}
    ycols = get_which_data_ycols(self, which_data_ycols)
    rows = get_which_data_rows(self, which_data_rows)

    X, X_variance, Y = get_x_y_var(self)
    free_dims = get_free_dims(self, visible_dims, None)
    
    plots = {}
    plots['dataplot'] = []
    plots['xerrorplot'] = []
    
    #one dimensional plotting
    if len(free_dims) == 1:
        for d in ycols:
            update_not_existing_kwargs(plot_kwargs, pl.defaults.data_1d)
            plots['dataplot'].append(pl.scatter(canvas, X[rows, free_dims], Y[rows, d], **plot_kwargs))
            if X_variance is not None:
                update_not_existing_kwargs(error_kwargs, pl.defaults.xerrorbar)
                plots['xerrorplot'].append(pl.xerrorbar(canvas, X[rows, free_dims].flatten(), Y[rows, d].flatten(),
                            2 * np.sqrt(X_variance[rows, free_dims].flatten()),
                            **error_kwargs))
    #2D plotting
    elif len(free_dims) == 2:
        for d in ycols:
            update_not_existing_kwargs(plot_kwargs, pl.defaults.data_2d)
            plots['dataplot'].append(pl.scatter(canvas, X[rows, free_dims[0]], X[rows, free_dims[1]], 
                                           c=Y[rows, d], vmin=Y.min(), vmax=Y.max(), **plot_kwargs))
    else:
        raise NotImplementedError("Cannot plot in more then two dimensions")
    return plots

def plot_data(self, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        error_kwargs=None, **plot_kwargs):
    """
    Plot the training data
      - For higher dimensions than two, use fixed_inputs to plot the data points with some of the inputs fixed.

    Can plot only part of the data
    using which_data_rows and which_data_ycols.

    :param which_data_rows: which of the training data to plot (default all)
    :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
    :param which_data_ycols: when the data has several columns (independant outputs), only plot these
    :type which_data_rows: 'all' or a list of integers
    :param visible_dims: an array specifying the input dimensions to plot (maximum two)
    :type visible_dims: a numpy array
    :param dict error_kwargs: kwargs for the error plot for the plotting library you are using
    :param kwargs plot_kwargs: kwargs for the data plot for the plotting library you are using
    
    :returns list: of plots created.
    """
    canvas, kwargs = pl.get_new_canvas(plot_kwargs)
    plots = _plot_data(self, canvas, which_data_rows, which_data_ycols, visible_dims, error_kwargs, **kwargs)
    return pl.show_canvas(canvas, plots)
