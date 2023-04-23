# #Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from matplotlib import pyplot as plt
import numpy as np

from .util import align_subplot_array, align_subplots

def ax_default(fignum, ax):
    if ax is None:
        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    return fig, ax

def meanplot(x, mu, color='#3300FF', ax=None, fignum=None, linewidth=2,**kw):
    _, axes = ax_default(fignum, ax)
    return axes.plot(x,mu,color=color,linewidth=linewidth,**kw)

def gpplot(x, mu, lower, upper, edgecol='#3300FF', fillcol='#33CCFF', ax=None, fignum=None, **kwargs):
    _, axes = ax_default(fignum, ax)

    mu = mu.flatten()
    x = x.flatten()
    lower = lower.flatten()
    upper = upper.flatten()

    plots = []

    #here's the mean
    plots.append(meanplot(x, mu, edgecol, axes))

    #here's the box
    kwargs['linewidth']=0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    plots.append(axes.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color=fillcol,**kwargs))

    #this is the edge:
    plots.append(meanplot(x, upper,color=edgecol, linewidth=0.2, ax=axes))
    plots.append(meanplot(x, lower,color=edgecol, linewidth=0.2, ax=axes))

    return plots

def gradient_fill(x, percentiles, ax=None, fignum=None, **kwargs):
    _, ax = ax_default(fignum, ax)

    plots = []

    #here's the box
    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 1./(len(percentiles))

    # pop where from kwargs
    where = kwargs.pop('where') if 'where' in kwargs else None
    # pop interpolate, which we actually do not do here!
    if 'interpolate' in kwargs: kwargs.pop('interpolate')

    def pairwise(inlist):
        l = len(inlist)
        for i in range(int(np.ceil(l/2.))):
            yield inlist[:][i], inlist[:][(l-1)-i]

    polycol = []
    for y1, y2 in pairwise(percentiles):
        import matplotlib.mlab as mlab
        # Handle united data, such as dates
        ax._process_unit_info(xdata=x, ydata=y1)
        ax._process_unit_info(ydata=y2)

        # Convert the arrays so we can work with them
        from numpy import ma
        x = ma.masked_invalid(ax.convert_xunits(x))
        y1 = ma.masked_invalid(ax.convert_yunits(y1))
        y2 = ma.masked_invalid(ax.convert_yunits(y2))

        if y1.ndim == 0:
            y1 = np.ones_like(x) * y1
        if y2.ndim == 0:
            y2 = np.ones_like(x) * y2

        if where is None:
            where = np.ones(len(x), bool)
        else:
            where = np.asarray(where, bool)

        if not (x.shape == y1.shape == y2.shape == where.shape):
            raise ValueError("Argument dimensions are incompatible")

        mask = reduce(ma.mask_or, [ma.getmask(a) for a in (x, y1, y2)])
        if mask is not ma.nomask:
            where &= ~mask

        polys = []
        for ind0, ind1 in mlab.contiguous_regions(where):
            xslice = x[ind0:ind1]
            y1slice = y1[ind0:ind1]
            y2slice = y2[ind0:ind1]

            if not len(xslice):
                continue

            N = len(xslice)
            X = np.zeros((2 * N + 2, 2), float)

            # the purpose of the next two lines is for when y2 is a
            # scalar like 0 and we want the fill to go all the way
            # down to 0 even if none of the y1 sample points do
            start = xslice[0], y2slice[0]
            end = xslice[-1], y2slice[-1]

            X[0] = start
            X[N + 1] = end

            X[1:N + 1, 0] = xslice
            X[1:N + 1, 1] = y1slice
            X[N + 2:, 0] = xslice[::-1]
            X[N + 2:, 1] = y2slice[::-1]

            polys.append(X)
        polycol.extend(polys)
    from matplotlib.collections import PolyCollection
    plots.append(PolyCollection(polycol, **kwargs))
    ax.add_collection(plots[-1], autolim=True)
    ax.autoscale_view()
    return plots

def gperrors(x, mu, lower, upper, edgecol=None, ax=None, fignum=None, **kwargs):
    _, axes = ax_default(fignum, ax)

    mu = mu.flatten()
    x = x.flatten()
    lower = lower.flatten()
    upper = upper.flatten()

    plots = []

    if edgecol is None:
        edgecol='#3300FF'

    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 1.


    if not 'lw' in kwargs.keys():
        kwargs['lw'] = 1.


    plots.append(axes.errorbar(x,mu,yerr=np.vstack([mu-lower,upper-mu]),color=edgecol,**kwargs))
    plots[-1][0].remove()
    return plots


def removeRightTicks(ax=None):
    ax = ax or plt.gca()
    for i, line in enumerate(ax.get_yticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)

def removeUpperTicks(ax=None):
    ax = ax or plt.gca()
    for i, line in enumerate(ax.get_xticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)

def fewerXticks(ax=None,divideby=2):
    ax = ax or plt.gca()
    ax.set_xticks(ax.get_xticks()[::divideby])

def x_frame1D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1] ==1, "x_frame1D is defined for one-dimensional inputs"
    if plot_limits is None:
        from ...core.parameterization.variational import VariationalPosterior
        if isinstance(X, VariationalPosterior):
            xmin,xmax = X.mean.min(0),X.mean.max(0)
        else:
            xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    else:
        raise ValueError("Bad limits for plotting")

    Xnew = np.linspace(xmin,xmax,resolution or 200)[:,None]
    return Xnew, xmin, xmax

def x_frame2D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1] ==2, "x_frame2D is defined for two-dimensional inputs"
    if plot_limits is None:
        xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    else:
        raise ValueError("Bad limits for plotting")

    resolution = resolution or 50
    xx,yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
    Xnew = np.vstack((xx.flatten(),yy.flatten())).T
    return Xnew, xx, yy, xmin, xmax
