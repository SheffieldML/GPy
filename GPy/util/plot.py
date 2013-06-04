# #Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import Tango
import pylab as pb
import numpy as np

def gpplot(x,mu,lower,upper,edgecol=Tango.colorsHex['darkBlue'],fillcol=Tango.colorsHex['lightBlue'],axes=None,**kwargs):
    if axes is None:
        axes = pb.gca()
    mu = mu.flatten()
    x = x.flatten()
    lower = lower.flatten()
    upper = upper.flatten()

    #here's the mean
    axes.plot(x,mu,color=edgecol,linewidth=2)

    #here's the box
    kwargs['linewidth']=0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    axes.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color=fillcol,**kwargs)

    #this is the edge:
    axes.plot(x,upper,color=edgecol,linewidth=0.2)
    axes.plot(x,lower,color=edgecol,linewidth=0.2)

def removeRightTicks(ax=None):
    ax = ax or pb.gca()
    for i, line in enumerate(ax.get_yticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)
def removeUpperTicks(ax=None):
    ax = ax or pb.gca()
    for i, line in enumerate(ax.get_xticklines()):
        if i%2 == 1:   # odd indices
            line.set_visible(False)
def fewerXticks(ax=None,divideby=2):
    ax = ax or pb.gca()
    ax.set_xticks(ax.get_xticks()[::divideby])

def align_subplots(N,M,xlim=None, ylim=None):
    """make all of the subplots have the same limits, turn off unnecessary ticks"""
    #find sensible xlim,ylim
    if xlim is None:
        xlim = [np.inf,-np.inf]
        for i in range(N*M):
            pb.subplot(N,M,i+1)
            xlim[0] = min(xlim[0],pb.xlim()[0])
            xlim[1] = max(xlim[1],pb.xlim()[1])
    if ylim is None:
        ylim = [np.inf,-np.inf]
        for i in range(N*M):
            pb.subplot(N,M,i+1)
            ylim[0] = min(ylim[0],pb.ylim()[0])
            ylim[1] = max(ylim[1],pb.ylim()[1])

    for i in range(N*M):
        pb.subplot(N,M,i+1)
        pb.xlim(xlim)
        pb.ylim(ylim)
        if (i)%M:
            pb.yticks([])
        else:
            removeRightTicks()
        if i<(M*(N-1)):
            pb.xticks([])
        else:
            removeUpperTicks()

def x_frame1D(X,plot_limits=None,resolution=None):
    """
    Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
    """
    assert X.shape[1] ==1, "x_frame1D is defined for one-dimensional inputs"
    if plot_limits is None:
        xmin,xmax = X.min(0),X.max(0)
        xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
    elif len(plot_limits)==2:
        xmin, xmax = plot_limits
    else:
        raise ValueError, "Bad limits for plotting"

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
        raise ValueError, "Bad limits for plotting"

    resolution = resolution or 50
    xx,yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
    Xnew = np.vstack((xx.flatten(),yy.flatten())).T
    return Xnew, xx, yy, xmin, xmax
