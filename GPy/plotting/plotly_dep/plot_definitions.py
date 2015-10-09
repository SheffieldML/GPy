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
from ..abstract_plotting_library import AbstractPlottingLibrary
from .. import Tango
from . import defaults
from plotly import tools
from plotly import plotly as py
from plotly.graph_objs import Scatter, Scatter3d, Line,\
    Marker, ErrorX, ErrorY, Bar, Heatmap, Trace,\
    Annotations, Annotation, Contour, Contours, Font, Surface
from plotly.exceptions import PlotlyDictKeyError

SYMBOL_MAP = {
    'o': 'dot',
    'v': 'triangle-down',
    '^': 'triangle-up',
    '<': 'triangle-left',
    '>': 'triangle-right',
    's': 'square',
    '+': 'cross',
    'x': 'x',
    '*': 'x',  # no star yet in plotly!!
    'D': 'diamond',
    'd': 'diamond',
}

class PlotlyPlots(AbstractPlottingLibrary):
    def __init__(self):
        super(PlotlyPlots, self).__init__()
        self._defaults = defaults.__dict__
        self.current_states = dict()
    
    def figure(self, rows=1, cols=1, specs=None, is_3d=False):
        if specs is None:
            specs = [[{'is_3d': is_3d}]*cols]*rows
        figure = tools.make_subplots(rows, cols, specs=specs)
        return figure
    
    def new_canvas(self, canvas=None, row=1, col=1, projection='2d', xlabel=None, ylabel=None, zlabel=None, title=None, xlim=None, ylim=None, zlim=None, **kwargs):
        #if 'filename' not in kwargs:
        #    print('PlotlyWarning: filename was not given, this may clutter your plotly workspace')
        #    filename = None
        #else: 
        #    filename = kwargs.pop('filename')
        if canvas is None:
            figure = self.figure(is_3d=projection=='3d')
            figure.layout.font = Font(family="Raleway, sans-serif")
        else:
            return canvas, kwargs
        return (figure, row, col), kwargs
    
    def add_to_canvas(self, canvas, traces, legend=False, **kwargs):
        figure, row, col = canvas
        def append_annotation(a, xref, yref):
            if 'xref' not in a:
                a['xref'] = xref
            if 'yref' not in a:
                a['yref'] = yref
            figure.layout.annotations.append(a)
        def append_trace(t, row, col):
            figure.append_trace(t, row, col)
        def recursive_append(traces):
            if isinstance(traces, Annotations):
                xref, yref = figure._grid_ref[row-1][col-1]
                for a in traces: 
                    append_annotation(a, xref, yref)
            elif isinstance(traces, (Trace)):
                try:
                    append_trace(traces, row, col)
                except PlotlyDictKeyError:
                    # Its a dictionary of plots:
                    for t in traces:
                        recursive_append(traces[t])
            elif isinstance(traces, (dict)):
                for t in traces:
                    recursive_append(traces[t])
            elif isinstance(traces, (tuple, list)):
                for t in traces:
                    recursive_append(t)
        recursive_append(traces)
        figure.layout['showlegend'] = legend
        return canvas
    
    def show_canvas(self, canvas, filename=None, **kwargs):
        figure, _, _ = canvas
        if len(figure.data) == 0:
            # add mock data
            figure.append_trace(Scatter(x=[], y=[], name='', showlegend=False), 1, 1)
        from ..gpy_plot.plot_util import in_ipynb
        if in_ipynb():
            py.iplot(figure, filename=filename)#self.current_states[hex(id(figure))]['filename'])
        else:
            py.plot(figure, filename=filename)#self.current_states[hex(id(figure))]['filename'])
        return figure
    
    def scatter(self, ax, X, Y, Z=None, color=Tango.colorsHex['mediumBlue'], cmap=None, label=None, marker='o', marker_kwargs=None, **kwargs):
        try:
            marker = SYMBOL_MAP[marker]
        except:
            #not matplotlib marker
            pass
        if Z is not None:
            return Scatter3d(x=X, y=Y, z=Z, mode='markers', showlegend=label is not None, marker=Marker(color=color, symbol=marker, colorscale=cmap, **marker_kwargs or {}), name=label, **kwargs)
        return Scatter(x=X, y=Y, mode='markers', showlegend=label is not None, marker=Marker(color=color, symbol=marker, colorscale=cmap, **marker_kwargs or {}), name=label, **kwargs)
    
    def plot(self, ax, X, Y, Z=None, color=None, label=None, line_kwargs=None, **kwargs):
        if 'mode' not in kwargs:
            kwargs['mode'] = 'lines'
        if Z is not None:
            return Scatter3d(x=X, y=Y, z=Z, showlegend=label is not None, line=Line(color=color, **line_kwargs or {}), name=label, **kwargs)
        return Scatter(x=X, y=Y, showlegend=label is not None, line=Line(color=color, **line_kwargs or {}), name=label, **kwargs)

    def plot_axis_lines(self, ax, X, Z=None, color=Tango.colorsHex['mediumBlue'], label=None, marker_kwargs=None, **kwargs):
        if X.shape[1] == 1:
            annotations = Annotations()
            for n, row in enumerate(X):
                annotations.append(
                    Annotation(
                        text='', 
                        x=row[0], y=0,
                        yref='paper',
                        ax=0, ay=20,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        showarrow=True))
        return annotations
        #if Z is not None:
        #    return Scatter3d(x=X[:,0], y=X[:,1], z=0, zref='paper', showlegend=label is not None, mode='markers', marker=Marker(color=color, symbol='diamond-tall', **marker_kwargs or {}), name=label, **kwargs)
        #return Scatter(x=X, y=0, mode='markers', showlegend=label is not None, marker=Marker(yref='paper', color=color, symbol='diamond-tall', **marker_kwargs or {}), name=label, **kwargs)
    
    def barplot(self, canvas, x, height, width=0.8, bottom=0, color=Tango.colorsHex['mediumBlue'], label=None, **kwargs):
        figure, _, _ = canvas
        if 'barmode' in kwargs:
            figure.layout['barmode'] = kwargs.pop('barmode')
        return Bar(x=x, y=height, marker=Marker(color=color), name=label)
        
    def xerrorbar(self, ax, X, Y, error, Z=None, color=Tango.colorsHex['mediumBlue'], label=None, error_kwargs=None, **kwargs):
        error_kwargs = error_kwargs or {}
        if (error.shape[0] == 2) and (error.ndim == 2):
            error_kwargs.update(dict(array=error[1], arrayminus=error[0], symmetric=False))
        else:
            error_kwargs.update(dict(array=error, symmetric=True))
        if Z is not None:
            return Scatter3d(x=X, y=Y, z=Z, mode='markers', 
                             error_x=ErrorX(color=color, **error_kwargs or {}), 
                             marker=Marker(size='0'), name=label, 
                             showlegend=label is not None, **kwargs)
        return Scatter(x=X, y=Y, mode='markers', 
                       error_x=ErrorX(color=color, **error_kwargs or {}), 
                       marker=Marker(size='0'), name=label, 
                      showlegend=label is not None, 
                       **kwargs)
        
    def yerrorbar(self, ax, X, Y, error, Z=None, color=Tango.colorsHex['mediumBlue'], label=None, error_kwargs=None, **kwargs):
        error_kwargs = error_kwargs or {}
        if (error.shape[0] == 2) and (error.ndim == 2):
            error_kwargs.update(dict(array=error[1], arrayminus=error[0], symmetric=False))
        else:
            error_kwargs.update(dict(array=error, symmetric=True))
        if Z is not None:
            return Scatter3d(x=X, y=Y, z=Z, mode='markers', 
                             error_y=ErrorY(color=color, **error_kwargs or {}), 
                             marker=Marker(size='0'), name=label, 
                             showlegend=label is not None, **kwargs)
        return Scatter(x=X, y=Y, mode='markers', 
                       error_y=ErrorY(color=color, **error_kwargs or {}), 
                       marker=Marker(size='0'), name=label, 
                      showlegend=label is not None, 
                       **kwargs)
    
    def imshow(self, ax, X, extent=None, label=None, vmin=None, vmax=None, **imshow_kwargs):
        if not 'showscale' in imshow_kwargs:
            imshow_kwargs['showscale'] = False
        return Heatmap(z=X, name=label, 
                       x0=extent[0], dx=float(extent[1]-extent[0])/(X.shape[0]-1),
                       y0=extent[2], dy=float(extent[3]-extent[2])/(X.shape[1]-1),
                       zmin=vmin, zmax=vmax, 
                       showlegend=label is not None, 
                       hoverinfo='z',
                       **imshow_kwargs)

    def imshow_interact(self, ax, plot_function, extent=None, label=None, resolution=None, vmin=None, vmax=None, **imshow_kwargs):
        # TODO stream interaction?
        super(PlotlyPlots, self).imshow_interact(ax, plot_function)

    def annotation_heatmap(self, ax, X, annotation, extent=None, label='Gradient', imshow_kwargs=None, **annotation_kwargs):
        imshow_kwargs.setdefault('label', label)
        imshow_kwargs.setdefault('showscale', True)
        imshow = self.imshow(ax, X, extent, **imshow_kwargs)
        X = X-X.min()
        X /= X.max()/2.
        X -= 1
        x = np.linspace(extent[0], extent[1], X.shape[0])
        y = np.linspace(extent[2], extent[3], X.shape[1])
        annotations = Annotations()
        for n, row in enumerate(annotation):
            for m, val in enumerate(row):
                var = X[n][m]
                annotations.append(
                    Annotation(
                        text=str(val), 
                        x=x[m], y=y[n],
                        xref='x1', yref='y1',
                        font=dict(color='white' if np.abs(var) > 0.8 else 'black', size=10),
                        opacity=.5,
                        showarrow=False))
        return imshow, annotations

    def annotation_heatmap_interact(self, ax, plot_function, extent, label=None, resolution=15, imshow_kwargs=None, **annotation_kwargs):
        super(PlotlyPlots, self).annotation_heatmap_interact(ax, plot_function, extent)
    
    def contour(self, ax, X, Y, C, levels=20, label=None, **kwargs):
        return Contour(x=X, y=Y, z=C, 
                       #ncontours=levels, contours=Contours(start=C.min(), end=C.max(), size=(C.max()-C.min())/levels), 
                       name=label, **kwargs)

    def surface(self, ax, X, Y, Z, color=None, label=None, **kwargs):
        return Surface(x=X, y=Y, z=Z, name=label, **kwargs)

    def fill_between(self, ax, X, lower, upper, color=Tango.colorsHex['mediumBlue'], label=None, line_kwargs=None, **kwargs):
        if not 'line' in kwargs:
            kwargs['line'] = Line(**line_kwargs or {})
        else:
            kwargs['line'].update(line_kwargs or {})
        if color.startswith('#'):
            fcolor = 'rgba ({c[0]}, {c[1]}, {c[2]}, {alpha})'.format(c=Tango.hex2rgb(color), alpha=kwargs.get('opacity', 1.0))
        else: fcolor = color
        u = Scatter(x=X, y=upper, fillcolor=fcolor, showlegend=label is not None, name=label, fill='tonexty', **kwargs)
        fcolor = '{}, {alpha})'.format(','.join(fcolor.split(',')[:-1]), alpha=0.0)
        l = Scatter(x=X, y=lower, fillcolor=fcolor, showlegend=False, fill='tonexty', name=label, **kwargs)
        return l, u 

    def fill_gradient(self, canvas, X, percentiles, color=Tango.colorsHex['mediumBlue'], label=None, **kwargs):
        ax = canvas
        plots = []
        
        if 'edgecolors' not in kwargs:
            kwargs['edgecolors'] = 'none'
        
        if 'facecolors' in kwargs:
            color = kwargs.pop('facecolors')
            
        if 'array' in kwargs:
            array = kwargs.pop('array')
        else:
            array = 1.-np.abs(np.linspace(-.97, .97, len(percentiles)-1))

        if 'alpha' in kwargs:
            alpha = kwargs.pop('alpha')
        else:
            alpha = .8

        if 'cmap' in kwargs:
            cmap = kwargs.pop('cmap')
        else:
            cmap = LinearSegmentedColormap.from_list('WhToColor', (color, color), N=array.size)
        cmap._init()
        cmap._lut[:-3, -1] = alpha*array

        kwargs['facecolors'] = [cmap(i) for i in np.linspace(0,1,cmap.N)]

        # pop where from kwargs
        where = kwargs.pop('where') if 'where' in kwargs else None
        # pop interpolate, which we actually do not do here!
        if 'interpolate' in kwargs: kwargs.pop('interpolate')

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            from itertools import tee
            #try:
            #    from itertools import izip as zip
            #except ImportError:
            #    pass
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)            
            
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
        
            from functools import reduce
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
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 0
        plots.append(PolyCollection(polycol, **kwargs))
        ax.add_collection(plots[-1], autolim=True)
        ax.autoscale_view()
        return plots
