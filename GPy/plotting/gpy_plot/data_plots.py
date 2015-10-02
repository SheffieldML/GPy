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
from . import update_not_existing_kwargs
from . import defaults

from functools import wraps
import numpy as np

def _plot_data(self, canvas, which_data_rows='all',
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
    """
    #deal with optional arguments
    if which_data_rows == 'all':
        which_data_rows = slice(None)
    if which_data_ycols == 'all':
        which_data_ycols = np.arange(self.output_dim)
    if error_kwargs is None:
        error_kwargs = {}

    if hasattr(self, 'has_uncertain_inputs') and self.has_uncertain_inputs():
        X = self.X.mean
        X_variance = self.X.variance
    else:
        X = self.X
        X_variance = None
    Y = self.Y

    #work out what the inputs are for plotting (1D or 2D)
    if visible_dims is None:
        visible_dims = np.arange(self.input_dim)
    assert visible_dims.size <= 2, "Visible inputs cannot be larger than two"
    free_dims = visible_dims
    
    #one dimensional plotting
    if len(free_dims) == 1:
        for d in which_data_ycols:
            update_not_existing_kwargs(plot_kwargs, defaults.data_1d)
            canvas.append(pl.scatter(canvas, X[which_data_rows, free_dims], Y[which_data_rows, d], **plot_kwargs))
            if X_variance is not None:
                update_not_existing_kwargs(error_kwargs, defaults.xerrorbar)
                canvas.append(pl.xerrorbar(canvas, X[which_data_rows, free_dims].flatten(), Y[which_data_rows, d].flatten(),
                            2 * np.sqrt(X_variance[which_data_rows, free_dims].flatten()),
                            **error_kwargs))
    #2D plotting
    elif len(free_dims) == 2:
        for d in which_data_ycols:
            update_not_existing_kwargs(plot_kwargs, defaults.data_2d)
            canvas = pl.scatter(canvas, X[which_data_rows, free_dims[0]], X[which_data_rows, free_dims[1]], 
                                           c=Y[which_data_rows, d], vmin=Y.min(), vmax=Y.max(), **plot_kwargs)
    else:
        raise NotImplementedError("Cannot plot in more then two dimensions")
    return canvas

@wraps(_plot_data)
def plot_data(self, which_data_rows='all',
        which_data_ycols='all', visible_dims=None,
        error_kwargs=None, **plot_kwargs):
    canvas, kwargs = pl.get_new_canvas(plot_kwargs)
    _plot_data(self, canvas, which_data_rows, which_data_ycols, visible_dims, error_kwargs, **kwargs)
    return pl.show_canvas(canvas)
