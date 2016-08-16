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
# * Neither the name of GPy.plotting.abstract_plotting_library nor the names of its
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

#===============================================================================
# Make sure that the necessary files and functions are
# defined in the plotting library:
class AbstractPlottingLibrary(object):
    def __init__(self):
        """
        Set the defaults dictionary in the _defaults variable:
        
        E.g. for matplotlib we define a file defaults.py and 
            set the dictionary of it here:
            
                from . import defaults
                _defaults = defaults.__dict__
        """
        self._defaults = {}
        self.__defaults = None
    
    @property
    def defaults(self):
        #===============================================================================
        if self.__defaults is None:
            from collections import defaultdict
            class defaultdict(defaultdict):
                def __getattr__(self, *args, **kwargs):
                    return defaultdict.__getitem__(self, *args, **kwargs)
            self.__defaults = defaultdict(dict, self._defaults)
        return self.__defaults
        #===============================================================================
    
    def figure(self, nrows, ncols, **kwargs):
        """
        Get a new figure with nrows and ncolumns subplots.
        Does not initialize the canvases yet.
        
        There is individual kwargs for the individual plotting libraries to use.
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")
    
    def new_canvas(self, figure=None, col=1, row=1, projection='2d', xlabel=None, ylabel=None, zlabel=None, title=None, xlim=None, ylim=None, zlim=None, **kwargs):
        """
        Return a canvas, kwargupdate for your plotting library. 

        if figure is not None, create a canvas in the figure
        at subplot position (col, row).
        
        This method does two things, it creates an empty canvas
        and updates the kwargs (deletes the unnecessary kwargs)
        for further usage in normal plotting.
        
        the kwargs are plotting library specific kwargs!

        :param {'2d'|'3d'} projection: The projection to use.

        E.g. in matplotlib this means it deletes references to ax, as
        plotting is done on the axis itself and is not a kwarg. 

        :param xlabel: the label to put on the xaxis
        :param ylabel: the label to put on the yaxis
        :param zlabel: the label to put on the zaxis (if plotting in 3d)
        :param title: the title of the plot
        :param legend: if True, plot a legend, if int make legend rows in the legend
        :param (float, float) xlim: the limits for the xaxis
        :param (float, float) ylim: the limits for the yaxis
        :param (float, float) zlim: the limits for the zaxis (if plotting in 3d)
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def add_to_canvas(self, canvas, plots, legend=True, title=None, **kwargs):
        """
        Add plots is a dictionary with the plots as the 
        items or a list of plots as items to canvas.
        
        The kwargs are plotting library specific kwargs!
        
        E.g. in matplotlib this does not have to do anything to add stuff, but
        we set the legend and title.

        !This function returns the updated canvas!

        :param title: the title of the plot
        :param legend: whether to plot a legend or not
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def show_canvas(self, canvas, **kwargs):
        """
        Draw/Plot the canvas given.
        """
        raise NotImplementedError

    def plot(self, cavas, X, Y, Z=None, color=None, label=None, **kwargs):
        """
        Make a line plot from for Y on X (Y = f(X)) on the canvas.
        If Z is not None, plot in 3d!
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def plot_axis_lines(self, ax, X, color=None, label=None, **kwargs):
        """
        Plot lines at the bottom (lower boundary of yaxis) of the axis at input location X.
        
        If X is two dimensional, plot in 3d and connect the axis lines to the bottom of the Z axis.   
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def surface(self, canvas, X, Y, Z, color=None, label=None, **kwargs):
        """
        Plot a surface for 3d plotting for the inputs (X, Y, Z).
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def scatter(self, canvas, X, Y, Z=None, color=None, vmin=None, vmax=None, label=None, **kwargs):
        """
        Make a scatter plot between X and Y on the canvas given.
        
        the kwargs are plotting library specific kwargs!
        
        :param canvas: the plotting librarys specific canvas to plot on.
        :param array-like X: the inputs to plot.
        :param array-like Y: the outputs to plot.
        :param array-like Z: the Z level to plot (if plotting 3d).
        :param array-like c: the colorlevel for each point.
        :param float vmin: minimum colorscale
        :param float vmax: maximum colorscale
        :param kwargs: the specific kwargs for your plotting library
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def barplot(self, canvas, x, height, width=0.8, bottom=0, color=None, label=None, **kwargs):
        """
        Plot vertical bar plot centered at x with height 
        and width of bars. The y level is at bottom.
        
        the kwargs are plotting library specific kwargs!

        :param array-like x: the center points of the bars
        :param array-like height: the height of the bars
        :param array-like width: the width of the bars
        :param array-like bottom: the start y level of the bars
        :param kwargs: kwargs for the specific library you are using.
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def xerrorbar(self, canvas, X, Y, error, color=None, label=None, **kwargs):
        """
        Make an errorbar along the xaxis for points at (X,Y) on the canvas.
        if error is two dimensional, the lower error is error[:,0] and
        the upper error is error[:,1]
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def yerrorbar(self, canvas, X, Y, error, color=None, label=None, **kwargs):
        """
        Make errorbars along the yaxis on the canvas given.
        if error is two dimensional, the lower error is error[0, :] and
        the upper error is error[1, :]
                
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def imshow(self, canvas, X, extent=None, label=None, vmin=None, vmax=None, **kwargs):
        """
        Show the image stored in X on the canvas.
        
        The origin of the image show is (0,0), such that X[0,0] gets plotted at [0,0] of the image!
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def imshow_interact(self, canvas, plot_function, extent=None, label=None, vmin=None, vmax=None, **kwargs):
        """
        This function is optional!

        Create an imshow controller to stream 
        the image returned by the plot_function. There is an imshow controller written for 
        mmatplotlib, which updates the imshow on changes in axis.
                
        The origin of the image show is (0,0), such that X[0,0] gets plotted at [0,0] of the image!
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def annotation_heatmap(self, canvas, X, annotation, extent, label=None, **kwargs):
        """
        Plot an annotation heatmap. That is like an imshow, but
        put the text of the annotation inside the cells of the heatmap (centered).
        
        :param canvas: the canvas to plot on
        :param array-like annotation: the annotation labels for the heatmap
        :param [horizontal_min,horizontal_max,vertical_min,vertical_max] extent: the extent of where to place the heatmap
        :param str label: the label for the heatmap
        :return: a list of both the heatmap and annotation plots [heatmap, annotation], or the interactive update object (alone)
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def annotation_heatmap_interact(self, canvas, plot_function, extent, label=None, resolution=15, **kwargs):
        """
        if plot_function is not None, return an interactive updated
        heatmap, which updates on axis events, so that one can zoom in 
        and out and the heatmap gets updated. See the matplotlib implementation
        in matplot_dep.controllers.
        
        the plot_function returns a pair (X, annotation) to plot, when called with
        a new input X (which would be the grid, which is visible on the plot
        right now)

        :param canvas: the canvas to plot on
        :param array-like annotation: the annotation labels for the heatmap
        :param [horizontal_min,horizontal_max,vertical_min,vertical_max] extent: the extent of where to place the heatmap
        :param str label: the label for the heatmap
        :return: a list of both the heatmap and annotation plots [heatmap, annotation], or the interactive update object (alone)
        :param plot_function: the function, which generates new data for given input locations X
        :param int resolution: the resolution of the interactive plot redraw - this is only needed when giving a plot_function
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def contour(self, canvas, X, Y, C, Z=None, color=None, label=None, **kwargs):
        """
        Make a contour plot at (X, Y) with heights/colors stored in C on the canvas.
        
        if Z is not None: make 3d contour plot at (X, Y, Z) with heights/colors stored in C on the canvas.
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def fill_between(self, canvas, X, lower, upper, color=None, label=None, **kwargs):
        """
        Fill along the xaxis between lower and upper.
        
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def fill_gradient(self, canvas, X, percentiles, color=None, label=None, **kwargs):
        """
        Plot a gradient (in alpha values) for the given percentiles.
                        
        the kwargs are plotting library specific kwargs!
        """
        print("fill_gradient not implemented in this backend.")
