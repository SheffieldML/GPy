# ===============================================================================
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
# ===============================================================================
import contextlib

import numpy as np
from ..abstract_plotting_library import AbstractPlottingLibrary
from .. import Tango
from . import defaults
import plotly
from plotly.basedatatypes import BaseTraceType
from plotly.subplots import make_subplots
from plotly.graph_objs import (
    Scatter,
    Scatter3d,
    ErrorX,
    ErrorY,
    Bar,
    Heatmap,
    Contour,
    Surface,
)
from plotly.graph_objs.layout import Annotation
from plotly.exceptions import PlotlyError

SYMBOL_MAP = {
    "o": "circle-dot",
    "v": "triangle-down",
    "^": "triangle-up",
    "<": "triangle-left",
    ">": "triangle-right",
    "s": "square",
    "+": "cross",
    "x": "x",
    "*": "star",
    "D": "diamond",
    "d": "diamond",
}


class PlotlyPlotsBase(AbstractPlottingLibrary):
    def __init__(self):
        super(PlotlyPlotsBase, self).__init__()
        self._defaults = defaults.__dict__
        self.current_states = dict()

    def figure(self, rows=1, cols=1, specs=None, is_3d=False, **kwargs):
        if specs is None:
            specs = [[{"is_3d": is_3d}] * cols] * rows
        figure = make_subplots(rows, cols, specs=specs, **kwargs)
        return figure

    def new_canvas(
        self,
        figure=None,
        row=1,
        col=1,
        projection="2d",
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
        xlim=None,
        ylim=None,
        zlim=None,
        **kwargs,
    ):
        # if 'filename' not in kwargs:
        #    print('PlotlyWarning: filename was not given, this may clutter your plotly workspace')
        #    filename = None
        # else:
        #    filename = kwargs.pop('filename')
        if figure is None:
            figure = self.figure(is_3d=projection == "3d")
            figure.layout.font = dict(family="Raleway, sans-serif")
        if projection == "3d":
            figure.layout.legend.x = 0.5
            figure.layout.legend.bgcolor = "#DCDCDC"
        return (figure, row, col), kwargs

    def add_to_canvas(self, canvas, traces, legend=False, **kwargs):
        figure, row, col = canvas

        def append_annotation(a, xref, yref):
            # Plotly layout.annotations is an immutable tuple in recent versions.
            if isinstance(a, dict):
                payload = dict(a)
            elif hasattr(a, "to_plotly_json"):
                payload = a.to_plotly_json()
            else:
                payload = dict(a)
            payload.setdefault("xref", xref)
            payload.setdefault("yref", yref)
            existing = list(figure.layout.annotations or ())
            existing.append(payload)
            figure.layout.annotations = existing

        def append_trace(t, row, col):
            figure.add_trace(t, row=row, col=col)

        def recursive_append(traces):
            if isinstance(traces, Annotation):
                cell = figure._grid_ref[row - 1][col - 1]
                # Plotly ≥5 wraps SubplotRef in a 1-tuple; older returned (xref, yref).
                if isinstance(cell, tuple) and len(cell) == 1 and hasattr(cell[0], "trace_kwargs"):
                    cell = cell[0]
                if hasattr(cell, "trace_kwargs"):
                    xref = cell.trace_kwargs.get("xaxis", "x")
                    yref = cell.trace_kwargs.get("yaxis", "y")
                else:
                    xref, yref = cell
                append_annotation(traces, xref, yref)
            elif isinstance(traces, (Scatter, Scatter3d, ErrorX, ErrorY, Bar, Heatmap, Contour, Surface, BaseTraceType)):
                try:
                    append_trace(traces, row, col)
                except (PlotlyError, TypeError, ValueError):
                    # Dictionary of plots rather than a single trace:
                    for t in traces:
                        recursive_append(traces[t])
            elif isinstance(traces, (dict)):
                for t in traces:
                    recursive_append(traces[t])
            elif isinstance(traces, (tuple, list)):
                for t in traces:
                    recursive_append(t)

        recursive_append(traces)
        figure.layout["showlegend"] = legend
        return canvas

    def show_canvas(self, canvas, filename=None, **kwargs):
        return NotImplementedError

    def scatter(
        self,
        ax,
        X,
        Y,
        Z=None,
        color=Tango.colorsHex["mediumBlue"],
        cmap=None,
        label=None,
        marker="o",
        marker_kwargs=None,
        **kwargs,
    ):
        with contextlib.suppress(KeyError):
            # not matplotlib marker if missing from SYMBOL_MAP
            marker = SYMBOL_MAP[marker]
        marker_kwargs = marker_kwargs or {}
        if "symbol" not in marker_kwargs:
            marker_kwargs["symbol"] = marker
        X, Y = np.squeeze(X), np.squeeze(Y)
        if Z is not None:
            Z = np.squeeze(Z)
            return Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode="markers",
                showlegend=label is not None,
                marker=dict(color=color, colorscale=cmap, **marker_kwargs),
                name=label,
                **kwargs,
            )
        return Scatter(
            x=X,
            y=Y,
            mode="markers",
            showlegend=label is not None,
            marker=dict(color=color, colorscale=cmap, **marker_kwargs),
            name=label,
            **kwargs,
        )

    def plot(self, ax, X, Y, Z=None, color=None, label=None, line_kwargs=None, **kwargs):
        if "mode" not in kwargs:
            kwargs["mode"] = "lines"
        X, Y = np.squeeze(X), np.squeeze(Y)
        if Z is not None:
            Z = np.squeeze(Z)
            return Scatter3d(
                x=X,
                y=Y,
                z=Z,
                showlegend=label is not None,
                line=dict(color=color, **line_kwargs or {}),
                name=label,
                **kwargs,
            )
        return Scatter(
            x=X, y=Y, showlegend=label is not None, line=dict(color=color, **line_kwargs or {}), name=label, **kwargs
        )

    def plot_axis_lines(self, ax, X, color=Tango.colorsHex["mediumBlue"], label=None, marker_kwargs=None, **kwargs):
        if X.shape[1] == 1:
            annotations = []
            for _i, row in enumerate(X):
                annotations.append(
                    Annotation(
                        text="",
                        x=row[0],
                        y=0,
                        yref="paper",
                        ax=0,
                        ay=20,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=color,
                        showarrow=True,
                        # showlegend=i==0,
                        # label=label,
                    )
                )
            return annotations
        elif X.shape[1] == 2:
            marker_kwargs.setdefault("symbol", "diamond")
            opacity = kwargs.pop("opacity", 0.8)
            return Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=np.zeros(X.shape[0]),
                mode="markers",
                projection=dict(z=dict(show=True, opacity=opacity)),
                marker=dict(color=color, **marker_kwargs or {}),
                opacity=0,
                name=label,
                showlegend=label is not None,
                **kwargs,
            )

    def barplot(
        self, canvas, x, height, width=0.8, bottom=0, color=Tango.colorsHex["mediumBlue"], label=None, **kwargs
    ):
        figure, _, _ = canvas
        if "barmode" in kwargs:
            figure.layout["barmode"] = kwargs.pop("barmode")
        return Bar(x=x, y=height, marker=dict(color=color), name=label)

    def xerrorbar(
        self, ax, X, Y, error, Z=None, color=Tango.colorsHex["mediumBlue"], label=None, error_kwargs=None, **kwargs
    ):
        error_kwargs = error_kwargs or {}
        if (error.shape[0] == 2) and (error.ndim == 2):
            error_kwargs.update(dict(array=error[1], arrayminus=error[0], symmetric=False))
        else:
            error_kwargs.update(dict(array=error, symmetric=True))
        X, Y = np.squeeze(X), np.squeeze(Y)
        if Z is not None:
            Z = np.squeeze(Z)
            return Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode="markers",
                error_x=ErrorX(color=color, **error_kwargs or {}),
                marker=dict(size="0"),
                name=label,
                showlegend=label is not None,
                **kwargs,
            )
        return Scatter(
            x=X,
            y=Y,
            mode="markers",
            error_x=ErrorX(color=color, **error_kwargs or {}),
            marker=dict(size="0"),
            name=label,
            showlegend=label is not None,
            **kwargs,
        )

    def yerrorbar(
        self, ax, X, Y, error, Z=None, color=Tango.colorsHex["mediumBlue"], label=None, error_kwargs=None, **kwargs
    ):
        error_kwargs = error_kwargs or {}
        if (error.shape[0] == 2) and (error.ndim == 2):
            error_kwargs.update(dict(array=error[1], arrayminus=error[0], symmetric=False))
        else:
            error_kwargs.update(dict(array=error, symmetric=True))
        X, Y = np.squeeze(X), np.squeeze(Y)
        if Z is not None:
            Z = np.squeeze(Z)
            return Scatter3d(
                x=X,
                y=Y,
                z=Z,
                mode="markers",
                error_y=ErrorY(color=color, **error_kwargs or {}),
                marker=dict(size="0"),
                name=label,
                showlegend=label is not None,
                **kwargs,
            )
        return Scatter(
            x=X,
            y=Y,
            mode="markers",
            error_y=ErrorY(color=color, **error_kwargs or {}),
            marker=dict(size="0"),
            name=label,
            showlegend=label is not None,
            **kwargs,
        )

    def imshow(self, ax, X, extent=None, label=None, vmin=None, vmax=None, **imshow_kwargs):
        if "showscale" not in imshow_kwargs:
            imshow_kwargs["showscale"] = False
        return Heatmap(
            z=X,
            name=label,
            x0=extent[0],
            dx=float(extent[1] - extent[0]) / (X.shape[0] - 1),
            y0=extent[2],
            dy=float(extent[3] - extent[2]) / (X.shape[1] - 1),
            zmin=vmin,
            zmax=vmax,
            showlegend=label is not None,
            hoverinfo="z",
            **imshow_kwargs,
        )

    def imshow_interact(
        self, ax, plot_function, extent=None, label=None, resolution=None, vmin=None, vmax=None, **imshow_kwargs
    ):
        # TODO stream interaction?
        super(PlotlyPlotsBase, self).imshow_interact(ax, plot_function)

    def annotation_heatmap(
        self, ax, X, annotation, extent=None, label="Gradient", imshow_kwargs=None, **annotation_kwargs
    ):
        imshow_kwargs.setdefault("label", label)
        imshow_kwargs.setdefault("showscale", True)
        imshow = self.imshow(ax, X, extent, **imshow_kwargs)
        X = X - X.min()
        X /= X.max() / 2.0
        X -= 1
        x = np.linspace(extent[0], extent[1], X.shape[0])
        y = np.linspace(extent[2], extent[3], X.shape[1])
        annotations = []
        for n, row in enumerate(annotation):
            for m, val in enumerate(row):
                var = X[n][m]
                annotations.append(
                    Annotation(
                        text=str(val),
                        x=x[m],
                        y=y[n],
                        xref="x1",
                        yref="y1",
                        font=dict(color="white" if np.abs(var) > 0.8 else "black", size=10),
                        opacity=0.5,
                        showarrow=False,
                    )
                )
        return imshow, annotations

    def annotation_heatmap_interact(
        self, ax, plot_function, extent, label=None, resolution=15, imshow_kwargs=None, **annotation_kwargs
    ):
        super(PlotlyPlotsBase, self).annotation_heatmap_interact(ax, plot_function, extent)

    def contour(self, ax, X, Y, C, levels=20, label=None, **kwargs):
        return Contour(
            x=X,
            y=Y,
            z=C,
            # ncontours=levels, contours=Contours(start=C.min(), end=C.max(), size=(C.max()-C.min())/levels),
            name=label,
            **kwargs,
        )

    def surface(self, ax, X, Y, Z, color=None, label=None, **kwargs):
        return Surface(x=X, y=Y, z=Z, name=label, showlegend=label is not None, **kwargs)

    def fill_between(
        self, ax, X, lower, upper, color=Tango.colorsHex["mediumBlue"], label=None, line_kwargs=None, **kwargs
    ):
        if "line" not in kwargs:
            kwargs["line"] = dict(**line_kwargs or {})
        else:
            kwargs["line"].update(line_kwargs or {})
        if color.startswith("#"):
            fcolor = "rgba({c[0]}, {c[1]}, {c[2]}, {alpha})".format(
                c=Tango.hex2rgb(color), alpha=kwargs.get("opacity", 1.0)
            )
        else:
            fcolor = color
        u = Scatter(
            x=X,
            y=upper,
            fillcolor=fcolor,
            showlegend=label is not None,
            name=label,
            fill="tonextx",
            legendgroup=f"{label}_fill_({ax[1]},{ax[2]})",
            **kwargs,
        )
        # fcolor = '{}, {alpha})'.format(','.join(fcolor.split(',')[:-1]), alpha=0.0)
        l = Scatter(
            x=X,
            y=lower,
            fillcolor=fcolor,
            showlegend=False,
            name=label,
            legendgroup=f"{label}_fill_({ax[1]},{ax[2]})",
            **kwargs,
        )
        return l, u

    def fill_gradient(self, canvas, X, percentiles, color=Tango.colorsHex["mediumBlue"], label=None, **kwargs):
        if color.startswith("#"):
            colarray = Tango.hex2rgb(color)
            opacity = 0.9
        else:
            colarray = map(float(color.strip(")").split("(")[1]))
            if len(colarray) == 4:
                colarray, opacity = colarray[:3], colarray[3]

        alpha = opacity * (1.0 - np.abs(np.linspace(-1, 1, len(percentiles) - 1)))

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            from itertools import tee

            a, b = tee(iterable)
            next(b, None)
            return zip(a, b, strict=False)

        polycol = []
        for i, y1, a in zip(range(len(percentiles)), percentiles, alpha, strict=False):
            fcolor = "rgba({}, {}, {}, {alpha})".format(*colarray, alpha=a)
            if i == len(percentiles) / 2:
                polycol.append(
                    Scatter(
                        x=X,
                        y=y1,
                        fillcolor=fcolor,
                        showlegend=True,
                        name=label,
                        line=dict(width=0, smoothing=0),
                        mode="none",
                        fill="tonextx",
                        legendgroup="density",
                        hoverinfo="none",
                        **kwargs,
                    )
                )
            else:
                polycol.append(
                    Scatter(
                        x=X,
                        y=y1,
                        fillcolor=fcolor,
                        showlegend=False,
                        name=None,
                        line=dict(width=1, smoothing=0, color=fcolor),
                        mode="none",
                        fill="tonextx",
                        legendgroup="density",
                        hoverinfo="none",
                        **kwargs,
                    )
                )
        return polycol


class PlotlyPlotsOnline(PlotlyPlotsBase):
    def __init__(self):
        super(PlotlyPlotsOnline, self).__init__()

    def show_canvas(self, canvas, filename=None, **kwargs):
        figure, _, _ = canvas
        if len(figure.data) == 0:
            # add mock data
            figure.add_trace(Scatter(x=[], y=[], name="", showlegend=False), row=1, col=1)
        from ..gpy_plot.plot_util import in_ipynb

        if in_ipynb():
            return plotly.plotly.iplot(figure, filename=filename, **kwargs)
        else:
            return plotly.plotly.plot(
                figure, filename=filename, **kwargs
            )  # self.current_states[hex(id(figure))]['filename'])


class PlotlyPlotsOffline(PlotlyPlotsBase):
    def __init__(self):
        super(PlotlyPlotsOffline, self).__init__()

    def show_canvas(self, canvas, filename=None, **kwargs):
        figure, _, _ = canvas
        if len(figure.data) == 0:
            # add mock data
            figure.add_trace(Scatter(x=[], y=[], name="", showlegend=False), row=1, col=1)
        from ..gpy_plot.plot_util import in_ipynb

        if in_ipynb():
            plotly.offline.init_notebook_mode(connected=True)
            return plotly.offline.iplot(
                figure, filename=filename, **kwargs
            )  # self.current_states[hex(id(figure))]['filename'])
        else:
            return plotly.offline.plot(figure, filename=filename, **kwargs)
