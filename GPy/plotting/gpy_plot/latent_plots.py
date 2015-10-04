#===============================================================================
# Copyright (c) 2015, Max Zwiessele
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
# * Neither the name of GPy.plotting.gpy_plot.latent_plots nor the names of its
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
from . import pl
from .plot_util import get_x_y_var, get_free_dims, get_which_data_ycols,\
    get_which_data_rows, update_not_existing_kwargs, helper_predict_with_model,\
    helper_for_plot_data

def plot_prediction_fit(self, plot_limits=None,
        which_data_rows='all', which_data_ycols='all', 
        fixed_inputs=None, resolution=None,
        plot_raw=False, apply_link=False, visible_dims=None,
        predict_kw=None, scatter_kwargs=None, **plot_kwargs):
    """
    Plot the fit of the (Bayesian)GPLVM latent space prediction to the outputs.
    This scatters two output dimensions against each other and a line
    from the prediction in two dimensions between them.
    
    Give the Y_metadata in the predict_kw if you need it.
    
    :param which_data_rows: which of the training data to plot (default all)
    :type which_data_rows: 'all' or a slice object to slice self.X, self.Y
    :param array-like which_data_ycols: which columns of y to plot (array-like or list of ints)
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input dimension i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param int resolution: The resolution of the prediction [defaults are 1D:200, 2D:50]
    :param bool plot_raw: plot the latent function (usually denoted f) only?
    :param bool apply_link: whether to apply the link function of the GP to the raw prediction.
    :param array-like visible_dims: which columns of the input X (!) to plot (array-like or list of ints)
    :param dict predict_kw: the keyword arguments for the prediction. If you want to plot a specific kernel give dict(kern=<specific kernel>) in here
    :param dict sactter_kwargs: kwargs for the scatter plot, specific for the plotting library you are using
    :param kwargs plot_kwargs: kwargs for the data plot for the plotting library you are using
    """
    canvas, kwargs = pl.get_new_canvas(plot_kwargs)
    plots = _plot_prediction_fit(self, canvas, plot_limits, which_data_rows, which_data_ycols, 
                                 fixed_inputs, resolution, plot_raw, 
                                 apply_link, visible_dims,
                                 predict_kw, scatter_kwargs, **kwargs)
    return pl.show_canvas(canvas, plots)

def _plot_prediction_fit(self, canvas, plot_limits=None,
        which_data_rows='all', which_data_ycols='all', 
        fixed_inputs=None, resolution=None,
        plot_raw=False, apply_link=False, visible_dims=False,
        predict_kw=None, scatter_kwargs=None, **plot_kwargs):
    
    ycols = get_which_data_ycols(self, which_data_ycols)
    rows = get_which_data_rows(self, which_data_rows)

    if visible_dims is None:
        visible_dims = self.get_most_significant_input_dimensions()[:1]

    X, _, Y, _, free_dims, Xgrid, _, _, _, _, resolution = helper_for_plot_data(self, plot_limits, visible_dims, fixed_inputs, resolution)
    
    plots = {}
    
    if len(free_dims)<2:
        if len(free_dims)==1:
            if scatter_kwargs is None:
                scatter_kwargs = {}
            update_not_existing_kwargs(scatter_kwargs, pl.defaults.data_y_1d)  # @UndefinedVariable
            plots['output'] = pl.scatter(canvas, Y[rows, ycols[0]], Y[rows, ycols[1]],
                                      c=X[rows, free_dims[0]],
                                      **scatter_kwargs)
            if predict_kw is None:
                predict_kw = {}
            mu, _, _ = helper_predict_with_model(self, Xgrid, plot_raw, 
                                              apply_link, None, 
                                              ycols, predict_kw)
            update_not_existing_kwargs(plot_kwargs, pl.defaults.data_y_1d_plot)  # @UndefinedVariable
            plots['output_fit'] = pl.plot(canvas, mu[:, 0], mu[:, 1], **plot_kwargs)
        else:
            pass #Nothing to plot!
    else:
        raise NotImplementedError("Cannot plot in more then one dimension.")
    return plots
    
    
    
    