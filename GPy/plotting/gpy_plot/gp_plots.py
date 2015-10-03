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
from functools import wraps

from . import pl
from .plot_util import get_x_y_var, get_fixed_dims, get_free_dims, \
    x_frame1D, x_frame2D, update_not_existing_kwargs, \
    helper_predict_with_model

def _helper_for_plots(self, plot_limits, fixed_inputs, resolution):
    """
    Figure out the data, free_dims and create an Xgrid for
    the prediction. 
    """
    X, Xvar, Y = get_x_y_var(self)

    #work out what the inputs are for plotting (1D or 2D)
    fixed_dims = get_fixed_dims(self, fixed_inputs)
    free_dims = get_free_dims(self, None, fixed_dims)
    
    if len(free_dims) == 1:
        #define the frame on which to plot
        resolution = resolution or 200
        Xnew, xmin, xmax = x_frame1D(X[:,free_dims], plot_limits=plot_limits, resolution=resolution)
        Xgrid = np.empty((Xnew.shape[0],self.input_dim))
        Xgrid[:,free_dims] = Xnew
        for i,v in fixed_dims:
            Xgrid[:,i] = v
        x = Xgrid
        y = None
    elif len(free_dims) == 2:
        #define the frame for plotting on
        resolution = resolution or 50
        Xnew, x, y, xmin, xmax = x_frame2D(X[:,free_dims], plot_limits, resolution)
        Xgrid = np.empty((Xnew.shape[0],self.input_dim))
        Xgrid[:,free_dims] = Xnew
        for i,v in fixed_dims:
            Xgrid[:,i] = v    
    return X, Xvar, Y, fixed_dims, free_dims, Xgrid, x, y, xmin, xmax, resolution

def plot_mean(self, plot_limits=None, fixed_inputs=None,
              resolution=None, plot_raw=False,
              Y_metadata=None, apply_link=False, 
              which_data_ycols='all',
              levels=20,
              predict_kw=None,
              **kwargs):
    """
    Plot the mean of the GP.
    
    :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
    :type plot_limits: np.array
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input dimension i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param int resolution: The resolution of the prediction [defaults are 1D:200, 2D:50]
    :param bool plot_raw: plot the latent function (usually denoted f) only?
    :param dict Y_metadata: the Y_metadata (for e.g. heteroscedastic GPs)
    :param bool apply_link: whether to apply the link function of the GP to the raw prediction.
    :param array-like which_data_ycols: which columns of y to plot (array-like or list of ints)
    :param dict predict_kw: the keyword arguments for the prediction. If you want to plot a specific kernel give dict(kern=<specific kernel>) in here
    :param int levels: for 2D plotting, the number of contour levels to use is 
    """
    canvas, kwargs = pl.get_new_canvas(kwargs)
    plots = _plot_mean(self, canvas, plot_limits, fixed_inputs, resolution, plot_raw, Y_metadata, apply_link, which_data_ycols, levels, predict_kw, **kwargs)
    return pl.show_canvas(canvas, plots)

@wraps(plot_mean)
def _plot_mean(self, canvas, plot_limits=None, fixed_inputs=None,
              resolution=None, plot_raw=False,
              Y_metadata=None, apply_link=False, 
              which_data_ycols=None,
              levels=20, 
              predict_kw=None, **kwargs):
    if predict_kw is None:
        predict_kw = {}

    _, _, _, _, free_dims, Xgrid, x, y, _, _, resolution = _helper_for_plots(self, plot_limits, fixed_inputs, resolution)

    if len(free_dims<=2):
        which_data_ycols = get_which_data_ycols(self, which_data_ycols)
        mu, _ = helper_predict_with_model(self, Xgrid, plot_raw, apply_link, None, which_data_ycols, **predict_kw)
        if len(free_dims)==1:
            # 1D plotting:
            update_not_existing_kwargs(kwargs, pl.defaults.meanplot_1d)
            return dict(gpmean=[pl.plot(canvas, Xgrid, mu, **kwargs)])
        else:
            update_not_existing_kwargs(kwargs, pl.defaults.meanplot_2d)
            return dict(gpmean=[pl.contour(canvas, x, y, 
                                           mu.reshape(resolution, resolution), 
                                           levels=levels, **kwargs)])

def plot_confidence(self, lower=2.5, upper=97.5, plot_limits=None, fixed_inputs=None,
              resolution=None, plot_raw=False,
              Y_metadata=None, apply_link=False, 
              which_data_ycols='all',
              predict_kw=None, 
              **kwargs):
    """
    Plot the confidence interval between the percentiles lower and upper.
    E.g. the 95% confidence interval is $2.5, 97.5$.
    Note: Only implemented for one dimension!
    
    :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
    :type plot_limits: np.array
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input dimension i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param int resolution: The resolution of the prediction [default:200]
    :param bool plot_raw: plot the latent function (usually denoted f) only?
    :param dict Y_metadata: the Y_metadata (for e.g. heteroscedastic GPs)
    :param bool apply_link: whether to apply the link function of the GP to the raw prediction.
    :param array-like which_data_ycols: which columns of y to plot (array-like or list of ints)
    :param dict predict_kw: the keyword arguments for the prediction. If you want to plot a specific kernel give dict(kern=<specific kernel>) in here
    """
    canvas, kwargs = pl.get_new_canvas(kwargs)
    plots = _plot_confidence(self, canvas, lower, upper, plot_limits, 
                             fixed_inputs, resolution, plot_raw, Y_metadata, 
                             apply_link, which_data_ycols, 
                             predict_kw, **kwargs)
    return pl.show_canvas(canvas, plots)

def _plot_confidence(self, canvas, lower, upper, plot_limits=None, fixed_inputs=None,
              resolution=None, plot_raw=False,
              Y_metadata=None, apply_link=False, 
              which_data_ycols=None, 
              predict_kw=None, 
              **kwargs):
    if predict_kw is None:
        predict_kw = {}
    
    _, _, _, _, _, Xgrid, _, _, _, _, _ = _helper_for_plots(self, plot_limits, fixed_inputs, resolution)

    update_not_existing_kwargs(kwargs, pl.defaults.confidence_interval)
    _, percs = helper_predict_with_model(self, Xgrid, plot_raw, apply_link, (lower, upper), which_data_ycols, **predict_kw)

    return dict(gpconfidence=pl.fill_between(canvas, Xgrid, percs[0], percs[1], **kwargs))

    
    