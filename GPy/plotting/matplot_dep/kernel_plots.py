# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from matplotlib import pyplot as pb
from matplotlib.textpath import TextPath
from matplotlib.transforms import offset_copy
from .base_plots import ax_default

def add_bar_labels(fig, ax, bars, bottom=0):
    transOffset = offset_copy(ax.transData, fig=fig,
                              x=0., y= -2., units='points')
    transOffsetUp = offset_copy(ax.transData, fig=fig,
                              x=0., y=1., units='points')
    for bar in bars:
        for i, [patch, num] in enumerate(zip(bar.patches, np.arange(len(bar.patches)))):
            if len(bottom) == len(bar): b = bottom[i]
            else: b = bottom
            height = patch.get_height() + b
            xi = patch.get_x() + patch.get_width() / 2.
            va = 'top'
            c = 'w'
            t = TextPath((0, 0), "${xi}$".format(xi=xi), rotation=0, ha='center')
            transform = transOffset
            if patch.get_extents().height <= t.get_extents().height + 5:
                va = 'bottom'
                c = 'k'
                transform = transOffsetUp
            ax.text(xi, height, "${xi}$".format(xi=int(num)), color=c, rotation=0, ha='center', va=va, transform=transform)

    ax.set_xticks([])


def plot_bars(fig, ax, x, ard_params, color, name, bottom=0):
    return ax.bar(left=x, height=ard_params.view(np.ndarray), width=.8,
                  bottom=bottom, align='center',
                  color=color, edgecolor='k', linewidth=1.2,
                  label=name.replace("_"," "))




def plot(kernel,x=None, fignum=None, ax=None, title=None, plot_limits=None, resolution=None, **mpl_kwargs):
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
    _, ax = ax_default(fignum,ax)

    if title is None:
        ax.set_title('%s kernel' % kernel.name)
    else:
        ax.set_title(title)


    if kernel.input_dim == 1:
        if x is None:
            x = np.zeros((1, 1))
        else:
            x = np.asarray(x)
            assert x.size == 1, "The size of the fixed variable x is not 1"
            x = x.reshape((1, 1))

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
