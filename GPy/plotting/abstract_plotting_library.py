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
    
    def get_new_canvas(self, plot_3d=False, **kwargs):
        """
        Return a canvas, kwargupdate for your plotting library. 
        
        This method does two things, it creates an empty canvas
        and updates the kwargs (deletes the unnecessary kwargs)
        for further usage in normal plotting.
        
        the kwargs are plotting library specific kwargs!

        :param bool plot_3d: whether to plot in 3d.
        
        E.g. in matplotlib this means it deletes references to ax, as
        plotting is done on the axis itself and is not a kwarg. 
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def show_canvas(self, canvas, plots, xlabel=None, ylabel=None, zlabel=None, title=None, xlim=None, ylim=None, zlim=None, legend=True, **kwargs):
        """
        Show the canvas given. 
        plots is a dictionary with the plots
        as the items.
        
        the kwargs are plotting library specific kwargs!

        :param xlabel: the label to put on the xaxis
        :param ylabel: the label to put on the yaxis
        :param zlabel: the label to put on the zaxis (if plotting in 3d)
        :param title: the title of the plot
        :param (float, float) xlim: the limits for the xaxis
        :param (float, float) ylim: the limits for the yaxis
        :param (float, float) zlim: the limits for the zaxis (if plotting in 3d)
        :param legend: whether to put a legend on
        
        E.g. in matplotlib this does not have to do anything, we make the tight plot, though.
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

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

    def scatter(self, canvas, X, Y, Z=None, c=None, vmin=None, vmax=None, label=None, **kwargs):
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
        if error is two dimensional, the lower error is error[:,0] and
        the upper error is error[:,1]
                
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def imshow(self, canvas, X, label=None, color=None, **kwargs):
        """
        Show the image stored in X on the canvas/
                
        the kwargs are plotting library specific kwargs!
        """
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")

    def contour(self, canvas, X, Y, C, color=None, label=None, **kwargs):
        """
        Make a contour plot at (X, Y) with heights stored in C on the canvas.
                
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
        raise NotImplementedError("Implement all plot functions in AbstractPlottingLibrary in order to use your own plotting library")
