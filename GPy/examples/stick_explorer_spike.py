# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""CIP-000B spike: browser latent explorer for the OSU stick man.

This is not the public API. The follow-up task extracts ``GPy.plotting.explore``.

Jupyter::

    from GPy.examples.stick_explorer_spike import stick_explorer
    stick_explorer()

Standalone (from the repository root, conda env ``py311``)::

    panel serve GPy/examples/stick_explorer_spike.py --show

Requires HoloViews, Panel, Plotly, and Bokeh in the environment. They are not
GPy core dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _latent_values(model):
    """Latent locations: posterior mean for Bayesian GPLVM, ``X`` otherwise.

    Do not use ``hasattr(X, "mean")``: Param/ndarray ``mean`` is a method.
    """
    from GPy.core.parameterization.variational import VariationalPosterior

    X = model.X
    if isinstance(X, VariationalPosterior):
        return X.mean
    return X


def latent_mean_row(model):
    """Writable ``(1, Q)`` row from a GPLVM or BayesianGPLVM."""
    vals = _latent_values(model)
    row = np.array(np.asarray(vals[:1, :]), dtype=float, copy=True)
    return np.atleast_2d(row.reshape(row.shape[0], -1)[:1])


def latent_array(model):
    """``(N, Q)`` latent locations for the scatter plot."""
    return np.array(np.asarray(_latent_values(model)), dtype=float, copy=True)


def assemble_latent(mean_row, x, y, dim_x, dim_y):
    """Copy ``mean_row`` and write the two active coordinates."""
    z = np.array(np.asarray(mean_row), dtype=float, copy=True)
    z = np.atleast_2d(z.reshape(z.shape[0], -1)[:1])
    z[0, int(dim_x)] = float(x)
    z[0, int(dim_y)] = float(y)
    return z


def y_to_xyz(y):
    """Reshape a stick-man vector into an ``(J, 3)`` point cloud."""
    vec = np.asarray(y, dtype=float).reshape(-1)
    if vec.size % 3 != 0:
        raise ValueError("stick-man vector length must be divisible by 3")
    return vec.reshape(3, vec.size // 3).T


def stick_polylines(xyz, connect):
    """NaN-separated bone polylines from a joint connectivity matrix."""
    xyz = np.asarray(xyz, dtype=float)
    connect = np.asarray(connect)
    i_idx, j_idx = np.nonzero(connect)
    xs, ys, zs = [], [], []
    for i, j in zip(i_idx, j_idx, strict=False):
        xs.extend((xyz[i, 0], xyz[j, 0], np.nan))
        ys.extend((xyz[i, 1], xyz[j, 1], np.nan))
        zs.extend((xyz[i, 2], xyz[j, 2], np.nan))
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def stick_plotly_figure(y, connect):
    """Plotly 3D stick figure for a reconstructed observation."""
    import plotly.graph_objects as go

    xyz = y_to_xyz(y)
    xs, ys, zs = stick_polylines(xyz, connect)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"width": 8, "color": "#111111"},
            name="bones",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker={"size": 4, "color": "#d62728"},
            name="joints",
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 32, "b": 0},
        scene={"aspectmode": "data"},
        showlegend=False,
        height=450,
        width=450,
        title="reconstruction",
    )
    return fig


def load_stick_model(optimize=False):
    """OSU stick-man Bayesian GPLVM without importing matplotlib."""
    import GPy
    import pods
    from GPy.models import BayesianGPLVM

    data = pods.datasets.osu_run1()
    Q = 6
    kernel = GPy.kern.RBF(Q, lengthscale=np.repeat(0.5, Q), ARD=True)
    model = BayesianGPLVM(data["Y"], Q, init="PCA", num_inducing=20, kernel=kernel)
    model.data = data
    model.likelihood.variance = 0.001
    if optimize:
        model.optimize("bfgs", messages=False, max_iters=200)
    return model


def stick_explorer(model=None, connect=None, optimize=False):
    """Panel app: latent scatter + Plotly stick + two latent-axis selectors."""
    import holoviews as hv
    import panel as pn
    from holoviews import streams

    pn.extension("plotly")
    hv.extension("bokeh")

    if model is None:
        model = load_stick_model(optimize=optimize)
    if connect is None:
        connect = model.data["connect"]

    X = latent_array(model)
    Q = X.shape[1]
    current = latent_mean_row(model)
    dims = list(range(Q))
    dim_x = pn.widgets.Select(label="x axis (green)", options=dims, value=0)
    dim_y = pn.widgets.Select(label="y axis (red)", options=dims, value=1)
    status = pn.pane.Markdown("Move the mouse over the latent scatter.")

    pipe = streams.Pipe(data=np.column_stack((X[:, 0], X[:, 1])))

    def _scatter(data):
        return hv.Points(data, ["x", "y"]).opts(
            size=7,
            width=420,
            height=420,
            tools=["hover"],
            xlabel="latent x",
            ylabel="latent y",
            color="#4c78a8",
        )

    latent_dmap = hv.DynamicMap(_scatter, streams=[pipe])
    pointer = streams.PointerXY(source=latent_dmap, x=float(current[0, 0]), y=float(current[0, 1]))
    stick_pane = pn.pane.Plotly(stick_plotly_figure(model.predict(current)[0][0], connect), sizing_mode="fixed")

    def _rebuild_stick(x, y):
        if x is None or y is None:
            return
        z = assemble_latent(current, x, y, dim_x.value, dim_y.value)
        current[:] = z
        mu, _ = model.predict(z)
        stick_pane.object = stick_plotly_figure(mu[0], connect)
        status.object = (
            f"z[{dim_x.value}]={float(x):.3f}, z[{dim_y.value}]={float(y):.3f}  "
            f"(other dims held at the current row)"
        )

    def _on_pointer(**contents):
        _rebuild_stick(contents.get("x"), contents.get("y"))

    def _on_dims(*_events):
        i, j = int(dim_x.value), int(dim_y.value)
        pipe.send(np.column_stack((X[:, i], X[:, j])))
        _rebuild_stick(float(current[0, i]), float(current[0, j]))

    pointer.add_subscriber(_on_pointer)
    dim_x.param.watch(_on_dims, "value")
    dim_y.param.watch(_on_dims, "value")

    return pn.Column(
        pn.pane.Markdown(
            "### CIP-000B spike: stick man latent explorer\n"
            "Move in the scatter to reconstruct. Other latent dimensions stay "
            "on the current row. No matplotlib GUI backend is used."
        ),
        pn.Row(dim_x, dim_y),
        pn.Row(latent_dmap, stick_pane),
        status,
    )


# Jupyter: ``from GPy.examples.stick_explorer_spike import stick_explorer``.
# ``panel serve`` and ``python this.py`` use another ``__name__``.
if __name__ != "GPy.examples.stick_explorer_spike":
    import panel as pn

    pn.extension("plotly")
    print("CIP-000B spike: building stick explorer…", flush=True)
    stick_explorer().servable()
    print("CIP-000B spike: app is servable.", flush=True)
