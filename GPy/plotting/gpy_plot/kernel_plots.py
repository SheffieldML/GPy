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
from . import pl
from .. import Tango
from .plot_util import get_x_y_var,\
    update_not_existing_kwargs, \
    helper_for_plot_data, scatter_label_generator, subsample_X,\
    find_best_layout_for_subplots

def plot_ARD(kernel, filtering=None, **kwargs):
    """
    If an ARD kernel is present, plot a bar representation using matplotlib

    :param fignum: figure number of the plot
    :param filtering: list of names, which to use for plotting ARD parameters.
                      Only kernels which match names in the list of names in filtering
                      will be used for plotting.
    :type filtering: list of names to use for ARD plot
    """
    canvas, kwargs = pl.new_canvas(kwargs)

    Tango.reset()
    
    bars = []
    ard_params = np.atleast_2d(kernel.input_sensitivity(summarize=False))
    bottom = 0
    last_bottom = bottom

    x = np.arange(kernel.input_dim)

    if filtering is None:
        filtering = kernel.parameter_names(recursive=False)

    for i in range(ard_params.shape[0]):
        if kernel.parameters[i].name in filtering:
            c = Tango.nextMedium()
            bars.append(pl.barplot(canvas, x, ard_params[i,:], color=c, label=kernel.parameters[i].name, bottom=bottom))
            last_bottom = ard_params[i,:]
            bottom += last_bottom
        else:
            print("filtering out {}".format(kernel.parameters[i].name))

    plt.add_to_canvas()
    ax.set_xlim(-.5, kernel.input_dim - .5)
    add_bar_labels(fig, ax, [bars[-1]], bottom=bottom-last_bottom)

    return dict(barplots=bars)

def plot_covariance(kernel, x=None, label=None, plot_limits=None, visible_dims=None, resolution=None, projection=None, levels=20, **mpl_kwargs):
    """
    plot a kernel.
    :param x: the value to use for the other kernel argument (kernels are a function of two variables!)
    :param fignum: figure number of the plot
    :param ax: matplotlib axis to plot on
    :param title: the matplotlib title
    :param plot_limits: the range over which to plot the kernel
    :resolution: the resolution of the lines used in plotting
    :mpl_kwargs avalid keyword arguments to pass through to matplotlib (e.g. lw=7)
    """
    canvas, error_kwargs = pl.new_canvas(projection=projection, **error_kwargs)
    _, _, _, _, free_dims, Xgrid, x, y, _, _, resolution = helper_for_plot_data(kernel, plot_limits, visible_dims, None, resolution)

    if len(free_dims)<=2:
        if len(free_dims)==1:
            if x is None: x = np.zeros((1, 1))
            else:
                x = np.asarray(x)
                assert x.size == 1, "The size of the fixed variable x is not 1"
                x = x.reshape((1, 1))
            # 1D plotting:
            update_not_existing_kwargs(kwargs, pl.defaults.meanplot_1d)  # @UndefinedVariable
            plots = dict(covariance=[pl.plot(canvas, Xgrid[:, free_dims], mu, label=label, **kwargs)])
        else:
            if projection == '2d':
                update_not_existing_kwargs(kwargs, pl.defaults.meanplot_2d)  # @UndefinedVariable
                plots = dict(covariance=[pl.contour(canvas, x, y, 
                                               mu.reshape(resolution, resolution).T, 
                                               levels=levels, label=label, **kwargs)])
            elif projection == '3d':
                update_not_existing_kwargs(kwargs, pl.defaults.meanplot_3d)  # @UndefinedVariable
                plots = dict(covariance=[pl.surface(canvas, x, y, 
                                               mu.reshape(resolution, resolution).T, 
                                               label=label, 
                                               **kwargs)])
                
        return pl.add_to_canvas(canvas, plots)

    if kernel.input_dim == 1:

        if plot_limits == None:
            xmin, xmax = (x - 5).flatten(), (x + 5).flatten()
        elif len(plot_limits) == 2:
            xmin, xmax = plot_limits
        else:
            raise ValueError("Bad limits for plotting")

        Xnew = np.linspace(xmin, xmax, resolution or 201)[:, None]
        Kx = kernel.K(Xnew, x)
        ax.plot(Xnew, Kx, **mpl_kwargs)
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("x")
        ax.set_ylabel("k(x,%0.1f)" % x)

    elif kernel.input_dim == 2:
        if x is None:
            x = np.zeros((1, 2))
        else:
            x = np.asarray(x)
            assert x.size == 2, "The size of the fixed variable x is not 2"
            x = x.reshape((1, 2))

        if plot_limits is None:
            xmin, xmax = (x - 5).flatten(), (x + 5).flatten()
        elif len(plot_limits) == 2:
            xmin, xmax = plot_limits
        else:
            raise ValueError("Bad limits for plotting")


        resolution = resolution or 51
        xx, yy = np.mgrid[xmin[0]:xmax[0]:1j * resolution, xmin[1]:xmax[1]:1j * resolution]
        Xnew = np.vstack((xx.flatten(), yy.flatten())).T
        Kx = kernel.K(Xnew, x)
        Kx = Kx.reshape(resolution, resolution).T
        ax.contour(xx, yy, Kx, vmin=Kx.min(), vmax=Kx.max(), cmap=pb.cm.jet, **mpl_kwargs) # @UndefinedVariable
        ax.set_xlim(xmin[0], xmax[0])
        ax.set_ylim(xmin[1], xmax[1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("k(x1,x2 ; %0.1f,%0.1f)" % (x[0, 0], x[0, 1]))
    else:
        raise NotImplementedError("Cannot plot a kernel with more than two input dimensions")
    
    pass