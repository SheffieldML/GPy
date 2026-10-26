# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
import pytest

from GPy.examples.stick_explorer_spike import (
    assemble_latent,
    latent_array,
    latent_mean_row,
    stick_plotly_figure,
    stick_polylines,
    y_to_xyz,
)


def test_y_to_xyz_matches_visualize_reshape():
    y = np.arange(12, dtype=float)
    xyz = y_to_xyz(y)
    assert xyz.shape == (4, 3)
    np.testing.assert_array_equal(xyz, y.reshape(3, 4).T)


def test_assemble_latent_writes_two_axes_only():
    mean = np.array([[1.0, 2.0, 3.0, 4.0]])
    z = assemble_latent(mean, x=9.0, y=-1.0, dim_x=0, dim_y=2)
    np.testing.assert_allclose(z, [[9.0, 2.0, -1.0, 4.0]])
    np.testing.assert_allclose(mean, [[1.0, 2.0, 3.0, 4.0]])


def test_stick_polylines_insert_nans_between_bones():
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    connect = np.zeros((3, 3))
    connect[0, 1] = 1.0
    connect[1, 2] = 1.0
    xs, ys, zs = stick_polylines(xyz, connect)
    assert np.isnan(xs[2])
    assert xs[0] == 0.0 and xs[1] == 1.0


def test_latent_row_from_gplvm_is_writable_and_2d():
    import GPy

    np.random.seed(0)
    model = GPy.models.GPLVM(np.random.randn(8, 9), 2)
    row = latent_mean_row(model)
    assert row.shape == (1, 2)
    row[0, 0] = 99.0
    assert model.X[0, 0] != 99.0
    arr = latent_array(model)
    assert arr.shape == (8, 2)


def test_latent_row_from_bayesian_gplvm_uses_posterior_mean():
    import GPy

    np.random.seed(3)
    model = GPy.models.BayesianGPLVM(np.random.randn(12, 9), 4, num_inducing=5)
    row = latent_mean_row(model)
    assert row.shape == (1, 4)
    row[0, 0] = 99.0
    assert float(model.X.mean[0, 0]) != 99.0
    assert latent_array(model).shape == (12, 4)


def test_predict_from_assembled_latent():
    import GPy

    np.random.seed(1)
    Y = np.random.randn(10, 9)
    model = GPy.models.GPLVM(Y, 2)
    z = assemble_latent(latent_mean_row(model), 0.1, -0.2, 0, 1)
    mu, _ = model.predict(z)
    assert mu.shape == (1, 9)
    fig = stick_plotly_figure(mu[0], np.eye(3))
    assert fig.data


def test_stick_explorer_builds_without_serving():
    pytest.importorskip("holoviews")
    pytest.importorskip("panel")
    pytest.importorskip("plotly")
    import GPy
    from GPy.examples.stick_explorer_spike import stick_explorer

    np.random.seed(2)
    Y = np.random.randn(6, 9)
    model = GPy.models.GPLVM(Y, 3)
    model.data = {"connect": np.eye(3)}
    app = stick_explorer(model=model, connect=model.data["connect"])
    assert app is not None
    assert len(app) >= 3
