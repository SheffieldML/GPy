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
# * Neither the name of GPy.plotting.matplot_dep.plot_definitions nor the names of its
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
from matplotlib import pyplot as plt
from ..abstract_plotting_library import AbstractPlottingLibrary
from . import defaults
from matplotlib.colors import LinearSegmentedColormap

class MatplotlibPlots(AbstractPlottingLibrary):
    def __init__(self):
        super(MatplotlibPlots, self).__init__()
        self._defaults = defaults.__dict__
    
    def get_new_canvas(self, plot_3d=False, kwargs):
        if plot_3d:
            from matplotlib.mplot3d import Axis3D  # @UnusedImport
            pr = '3d'
        else: pr=None
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        elif 'num' in kwargs and 'figsize' in kwargs:
            ax = plt.figure(num=kwargs.pop('num'), figsize=kwargs.pop('figsize')).add_subplot(111, projection=pr) 
        elif 'num' in kwargs:
            ax = plt.figure(num=kwargs.pop('num')).add_subplot(111, projection=pr)
        elif 'figsize' in kwargs:
            ax = plt.figure(figsize=kwargs.pop('figsize')).add_subplot(111, projection=pr)
        else:
            ax = plt.figure().add_subplot(111, projection=pr)
        # Add ax to kwargs to add all subsequent plots to this axis:
        #kwargs['ax'] = ax
        return ax, kwargs
    
    def show_canvas(self, ax, plots, xlabel=None, ylabel=None, 
                    zlabel=None, title=None, xlim=None, ylim=None, 
                    zlim=None, legend=True, **kwargs):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if zlabel is not None:
            ax.set_zlabel(zlabel)
        
        ax.set_title(title)
        
        try:
            ax.autoscale_view()
            ax.figure.canvas.draw()
            ax.figure.tight_layout()
        except:
            pass
        return plots
    
    def scatter(self, ax, X, Y, color=None, label=None, **kwargs):
        return ax.scatter(X, Y, c=color, label=label, **kwargs)
    
    def plot(self, ax, X, Y, color=None, label=None, **kwargs):
        return ax.plot(X, Y, color=color, label=label, **kwargs)

    def plot_axis_lines(self, ax, X, color=None, label=None, **kwargs):
        from matplotlib import transforms
        from matplotlib.path import Path
        if 'transform' not in kwargs:
            kwargs['transform'] = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        if 'marker' not in kwargs:
            kwargs['marker'] = Path([[-.2,0.],    [-.2,.5],    [0.,1.],    [.2,.5],     [.2,0.],     [-.2,0.]],
                                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
        return ax.scatter(X, np.zeros_like(X), c=color, label=label, **kwargs)

    def barplot(self, ax, x, height, width=0.8, bottom=0, color=None, label=None, **kwargs):
        if 'align' not in kwargs:
            kwargs['align'] = 'center'
        return ax.bar(left=x, height=height, width=width,
               bottom=bottom, label=label, color=color,  
               **kwargs)
        
    def xerrorbar(self, ax, X, Y, error, color=None, label=None, **kwargs):
        if not('linestyle' in kwargs or 'ls' in kwargs):
            kwargs['ls'] = 'none' 
        return ax.errorbar(X, Y, xerr=error, ecolor=color, label=label, **kwargs)
    
    def yerrorbar(self, ax, X, Y, error, color=None, label=None, **kwargs):
        if not('linestyle' in kwargs or 'ls' in kwargs):
            kwargs['ls'] = 'none'
        return ax.errorbar(X, Y, yerr=error, ecolor=color, label=label, **kwargs)
    
    def imshow(self, ax, X, label=None, **kwargs):
        return ax.imshow(X, label=label, **kwargs)
    
    def contour(self, ax, X, Y, C, levels=20, label=None, **kwargs):
        return ax.contour(X, Y, C, levels=np.linspace(C.min(), C.max(), levels), label=label, **kwargs)

    def fill_between(self, ax, X, lower, upper, color=None, label=None, **kwargs):
        return ax.fill_between(X, lower, upper, facecolor=color, label=label, **kwargs)

    def fill_gradient(self, canvas, X, percentiles, color=None, label=None, **kwargs):
        ax = canvas
        plots = []
        
        if 'edgecolors' not in kwargs:
            kwargs['edgecolors'] = 'none'
        
        if 'facecolors' not in kwargs:
            kwargs['facecolors'] = color
            
        if 'facecolors' in kwargs:
            kwargs['facecolor'] = kwargs.pop('facecolors')
            
        if 'cmap' not in kwargs:
            kwargs['cmap'] = LinearSegmentedColormap.from_list('WhToColor', ((1., 1., 1.), kwargs['facecolor']), N=len(percentiles)-1)
            kwargs['cmap']._init()
            
        if 'alpha' in kwargs:
            kwargs['cmap']._lut[:, -1] = kwargs['alpha']
        
        if 'array' not in kwargs:
            if (len(percentiles)%2) == 0:
                up = np.linspace(0, 1, len(percentiles)/2)
                kwargs['array'] = np.r_[up, up[::-1][1:]]
            else:
                up = np.linspace(0, 1, len(percentiles)/2)
                kwargs['array'] = np.r_[up, up[::-1]]

        # pop where from kwargs
        where = kwargs.pop('where') if 'where' in kwargs else None
        # pop interpolate, which we actually do not do here!
        if 'interpolate' in kwargs: kwargs.pop('interpolate')
        
        from itertools import tee, izip
        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return izip(a, b)            
            
        polycol = []
        for y1, y2 in pairwise(percentiles):
            import matplotlib.mlab as mlab
            # Handle united data, such as dates
            ax._process_unit_info(xdata=X, ydata=y1)
            ax._process_unit_info(ydata=y2)
            # Convert the arrays so we can work with them
            from numpy import ma
            x = ma.masked_invalid(ax.convert_xunits(X))
            y1 = ma.masked_invalid(ax.convert_yunits(y1))
            y2 = ma.masked_invalid(ax.convert_yunits(y2))
        
            if y1.ndim == 0:
                y1 = np.ones_like(x) * y1
            if y2.ndim == 0:
                y2 = np.ones_like(x) * y2
        
            if where is None:
                where = np.ones(len(x), np.bool)
            else:
                where = np.asarray(where, np.bool)
        
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
                p = np.zeros((2 * N + 2, 2), np.float)
        
                # the purpose of the next two lines is for when y2 is a
                # scalar like 0 and we want the fill to go all the way
                # down to 0 even if none of the y1 sample points do
                start = xslice[0], y2slice[0]
                end = xslice[-1], y2slice[-1]
        
                p[0] = start
                p[N + 1] = end
        
                p[1:N + 1, 0] = xslice
                p[1:N + 1, 1] = y1slice
                p[N + 2:, 0] = xslice[::-1]
                p[N + 2:, 1] = y2slice[::-1]
        
                polys.append(p)
            polycol.extend(polys)
        from matplotlib.collections import PolyCollection
        plots.append(PolyCollection(polycol, **kwargs))
        ax.add_collection(plots[-1], autolim=True)
        ax.autoscale_view()
        return plots
