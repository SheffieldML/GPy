# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
try:
    from GPy.plotting import Tango
    from matplotlib import pyplot as pb
except:
    pass


def plot_mapping(self, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20, samples=0, fignum=None, ax=None, fixed_inputs=[], linecol=Tango.colorsHex['darkBlue']):
    """
    Plots the mapping associated with the model.
      - In one dimension, the function is plotted.
      - In two dimsensions, a contour-plot shows the function
      - In higher dimensions, we've not implemented this yet !TODO!

    Can plot only part of the data and part of the posterior functions
    using which_data and which_functions

    :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
    :type plot_limits: np.array
    :param which_data: which if the training data to plot (default all)
    :type which_data: 'all' or a slice object to slice self.X, self.Y
    :param which_parts: which of the kernel functions to plot (additively)
    :type which_parts: 'all', or list of bools
    :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
    :type resolution: int
    :param levels: number of levels to plot in a contour plot.
    :type levels: int
    :param samples: the number of a posteriori samples to plot
    :type samples: int
    :param fignum: figure to plot on.
    :type fignum: figure number
    :param ax: axes to plot on.
    :type ax: axes handle
    :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
    :type fixed_inputs: a list of tuples
    :param linecol: color of line to plot.
    :type linecol:
    :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure

    """
    # TODO include samples
    if which_data == 'all':
        which_data = slice(None)

    if ax is None:
        fig = pb.figure(num=fignum)
        ax = fig.add_subplot(111)

    plotdims = self.input_dim - len(fixed_inputs)
    from ..gpy_plot.plot_util import x_frame1D, x_frame2D

    if plotdims == 1:

        Xu = self.X * self._Xscale + self._Xoffset # NOTE self.X are the normalized values now

        fixed_dims = np.array([i for i,v in fixed_inputs])
        freedim = np.setdiff1d(np.arange(self.input_dim),fixed_dims)

        Xnew, xmin, xmax = x_frame1D(Xu[:,freedim], plot_limits=plot_limits)
        Xgrid = np.empty((Xnew.shape[0],self.input_dim))
        Xgrid[:,freedim] = Xnew
        for i,v in fixed_inputs:
            Xgrid[:,i] = v

        f = self.predict(Xgrid, which_parts=which_parts)
        for d in range(y.shape[1]):
            ax.plot(Xnew, f[:, d], edgecol=linecol)

    elif self.X.shape[1] == 2:
        resolution = resolution or 50
        Xnew, _, _, xmin, xmax = x_frame2D(self.X, plot_limits, resolution)
        x, y = np.linspace(xmin[0], xmax[0], resolution), np.linspace(xmin[1], xmax[1], resolution)
        f = self.predict(Xnew, which_parts=which_parts)
        m = m.reshape(resolution, resolution).T
        ax.contour(x, y, f, levels, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet) # @UndefinedVariable
        ax.set_xlim(xmin[0], xmax[0])
        ax.set_ylim(xmin[1], xmax[1])

    else:
        raise NotImplementedError("Cannot define a frame with more than two input dimensions")
