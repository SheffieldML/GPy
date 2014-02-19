# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
import pylab as pb
import Tango
from matplotlib.textpath import TextPath
from matplotlib.transforms import offset_copy
from ...kern import Linear


def plot_ARD(kernel, fignum=None, ax=None, title='', legend=False):
    """If an ARD kernel is present, plot a bar representation using matplotlib

    :param fignum: figure number of the plot
    :param ax: matplotlib axis to plot on
    :param title:
        title of the plot,
        pass '' to not print a title
        pass None for a generic title
    """
    if ax is None:
        fig = pb.figure(fignum)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    Tango.reset()
    xticklabels = []
    bars = []
    x0 = 0
    for p in kernel._parameters_:
        c = Tango.nextMedium()
        if hasattr(p, 'ARD') and p.ARD:
            if title is None:
                ax.set_title('ARD parameters, %s kernel' % p.name)
            else:
                ax.set_title(title)
            if isinstance(p, Linear):
                ard_params = p.variances
            else:
                ard_params = 1. / p.lengthscale

            x = np.arange(x0, x0 + len(ard_params))
            bars.append(ax.bar(x, ard_params, align='center', color=c, edgecolor='k', linewidth=1.2, label=p.name.replace("_"," ")))
            xticklabels.extend([r"$\mathrm{{{name}}}\ {x}$".format(name=p.name, x=i) for i in np.arange(len(ard_params))])
            x0 += len(ard_params)
    x = np.arange(x0)
    transOffset = offset_copy(ax.transData, fig=fig,
                              x=0., y= -2., units='points')
    transOffsetUp = offset_copy(ax.transData, fig=fig,
                              x=0., y=1., units='points')
    for bar in bars:
        for patch, num in zip(bar.patches, np.arange(len(bar.patches))):
            height = patch.get_height()
            xi = patch.get_x() + patch.get_width() / 2.
            va = 'top'
            c = 'w'
            t = TextPath((0, 0), "${xi}$".format(xi=xi), rotation=0, usetex=True, ha='center')
            transform = transOffset
            if patch.get_extents().height <= t.get_extents().height + 3:
                va = 'bottom'
                c = 'k'
                transform = transOffsetUp
            ax.text(xi, height, "${xi}$".format(xi=int(num)), color=c, rotation=0, ha='center', va=va, transform=transform)
    # for xi, t in zip(x, xticklabels):
    #    ax.text(xi, maxi / 2, t, rotation=90, ha='center', va='center')
    # ax.set_xticklabels(xticklabels, rotation=17)
    ax.set_xticks([])
    ax.set_xlim(-.5, x0 - .5)
    if legend:
        if title is '':
            mode = 'expand'
            if len(bars) > 1:
                mode = 'expand'
            ax.legend(bbox_to_anchor=(0., 1.02, 1., 1.02), loc=3,
                      ncol=len(bars), mode=mode, borderaxespad=0.)
            fig.tight_layout(rect=(0, 0, 1, .9))
        else:
            ax.legend()
    return ax


def plot(kernel, x=None, plot_limits=None, which_parts='all', resolution=None, *args, **kwargs):
    if which_parts == 'all':
        which_parts = [True] * kernel.size
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
            raise ValueError, "Bad limits for plotting"

        Xnew = np.linspace(xmin, xmax, resolution or 201)[:, None]
        Kx = kernel.K(Xnew, x, which_parts)
        pb.plot(Xnew, Kx, *args, **kwargs)
        pb.xlim(xmin, xmax)
        pb.xlabel("x")
        pb.ylabel("k(x,%0.1f)" % x)

    elif kernel.input_dim == 2:
        if x is None:
            x = np.zeros((1, 2))
        else:
            x = np.asarray(x)
            assert x.size == 2, "The size of the fixed variable x is not 2"
            x = x.reshape((1, 2))

        if plot_limits == None:
            xmin, xmax = (x - 5).flatten(), (x + 5).flatten()
        elif len(plot_limits) == 2:
            xmin, xmax = plot_limits
        else:
            raise ValueError, "Bad limits for plotting"

        resolution = resolution or 51
        xx, yy = np.mgrid[xmin[0]:xmax[0]:1j * resolution, xmin[1]:xmax[1]:1j * resolution]
        xg = np.linspace(xmin[0], xmax[0], resolution)
        yg = np.linspace(xmin[1], xmax[1], resolution)
        Xnew = np.vstack((xx.flatten(), yy.flatten())).T
        Kx = kernel.K(Xnew, x, which_parts)
        Kx = Kx.reshape(resolution, resolution).T
        pb.contour(xg, yg, Kx, vmin=Kx.min(), vmax=Kx.max(), cmap=pb.cm.jet, *args, **kwargs) # @UndefinedVariable
        pb.xlim(xmin[0], xmax[0])
        pb.ylim(xmin[1], xmax[1])
        pb.xlabel("x1")
        pb.ylabel("x2")
        pb.title("k(x1,x2 ; %0.1f,%0.1f)" % (x[0, 0], x[0, 1]))
    else:
        raise NotImplementedError, "Cannot plot a kernel with more than two input dimensions"
