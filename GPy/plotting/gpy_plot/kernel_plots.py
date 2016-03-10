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
# * Neither the name of GPy.plotting.gpy_plot.kernel_plots nor the names of its
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
from .. import Tango
from .plot_util import update_not_existing_kwargs, helper_for_plot_data
from ...kern.src.kern import Kern, CombinationKernel

def plot_ARD(kernel, filtering=None, legend=False, canvas=None, **kwargs):
    """
    If an ARD kernel is present, plot a bar representation using matplotlib

    :param fignum: figure number of the plot
    :param filtering: list of names, which to use for plotting ARD parameters.
                      Only kernels which match names in the list of names in filtering
                      will be used for plotting.
    :type filtering: list of names to use for ARD plot
    """
    Tango.reset()

    ard_params = np.atleast_2d(kernel.input_sensitivity(summarize=False))
    bottom = 0
    last_bottom = bottom

    x = np.arange(kernel._effective_input_dim)

    parts = []
    def visit(x):
        if (not isinstance(x, CombinationKernel)) and isinstance(x, Kern):
            parts.append(x)
    kernel.traverse(visit)

    if filtering is None:
        filtering = [k.name for k in parts]

    bars = []
    kwargs = update_not_existing_kwargs(kwargs, pl().defaults.ard)


    if canvas is None:
        canvas, kwargs = pl().new_canvas(xlim=(-.5, kernel._effective_input_dim-.5), xlabel='input dimension', ylabel='sensitivity', **kwargs)

    for i in range(ard_params.shape[0]):
        if parts[i].name in filtering:
            c = Tango.nextMedium()
            bars.append(pl().barplot(canvas, x,
                                     ard_params[i,:], color=c,
                                     label=parts[i].name,
                                     bottom=bottom, **kwargs))
            last_bottom = ard_params[i,:]
            bottom += last_bottom
        else:
            print("filtering out {}".format(parts[i].name))

    #add_bar_labels(fig, ax, [bars[-1]], bottom=bottom-last_bottom)

    return pl().add_to_canvas(canvas, bars, legend=legend)

def plot_covariance(kernel, x=None, label=None,
             plot_limits=None, visible_dims=None, resolution=None,
             projection='2d', levels=20, **kwargs):
    """
    Plot a kernel covariance w.r.t. another x.

    :param array-like x: the value to use for the other kernel argument (kernels are a function of two variables!)
    :param plot_limits: the range over which to plot the kernel
    :type plot_limits: Either (xmin, xmax) for 1D or (xmin, xmax, ymin, ymax) / ((xmin, xmax), (ymin, ymax)) for 2D
    :param array-like visible_dims: input dimensions (!) to use for x. Make sure to select 2 or less dimensions to plot.
    :resolution: the resolution of the lines used in plotting. for 2D this defines the grid for kernel evaluation.
    :param {2d|3d} projection: What projection shall we use to plot the kernel?
    :param int levels: for 2D projection, how many levels for the contour plot to use?
    :param kwargs:  valid kwargs for your specific plotting library
    """
    X = np.ones((2, kernel._effective_input_dim)) * [[-3], [3]]
    _, free_dims, Xgrid, xx, yy, _, _, resolution = helper_for_plot_data(kernel, X, plot_limits, visible_dims, None, resolution)

    from numbers import Number
    if x is None:
        from ...kern.src.stationary import Stationary
        x = np.ones((1, kernel._effective_input_dim)) * (not isinstance(kernel, Stationary))
    elif isinstance(x, Number):
        x = np.ones((1, kernel._effective_input_dim))*x
    K = kernel.K(Xgrid, x)

    if projection == '3d':
        xlabel = 'X[:,0]'
        ylabel = 'X[:,1]'
        zlabel = "k(X, {!s})".format(np.asanyarray(x).tolist())
    else:
        xlabel = 'X'
        ylabel = "k(X, {!s})".format(np.asanyarray(x).tolist())
        zlabel = None

    canvas, kwargs = pl().new_canvas(projection=projection, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

    if len(free_dims)<=2:
        if len(free_dims)==1:
            # 1D plotting:
            update_not_existing_kwargs(kwargs, pl().defaults.meanplot_1d)  # @UndefinedVariable
            plots = dict(covariance=[pl().plot(canvas, Xgrid[:, free_dims], K, label=label, **kwargs)])
        else:
            if projection == '2d':
                update_not_existing_kwargs(kwargs, pl().defaults.meanplot_2d)  # @UndefinedVariable
                plots = dict(covariance=[pl().contour(canvas, xx[:, 0], yy[0, :],
                                               K.reshape(resolution, resolution),
                                               levels=levels, label=label, **kwargs)])
            elif projection == '3d':
                update_not_existing_kwargs(kwargs, pl().defaults.meanplot_3d)  # @UndefinedVariable
                plots = dict(covariance=[pl().surface(canvas, xx, yy,
                                               K.reshape(resolution, resolution),
                                               label=label,
                                               **kwargs)])
        return pl().add_to_canvas(canvas, plots)

    else:
        raise NotImplementedError("Cannot plot a kernel with more than two input dimensions")
