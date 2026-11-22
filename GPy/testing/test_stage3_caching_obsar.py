# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""CIP-000F Stage 3: version-counter cache and ndarray X/Y."""

import io
import pickle
import sys

import numpy as np

import GPy
from GPy.core.gp import _as_data_array
from GPy.core.parameterization import ObsAr
from GPy.core.parameterization.caching import Cache_this, FunctionCache


def test_gp_xy_are_plain_ndarray_not_obsar():
    np.random.seed(0)
    X = np.random.randn(15, 1)
    Y = np.sin(X) + 0.05 * np.random.randn(15, 1)
    m = GPy.models.GPRegression(X, Y)
    assert isinstance(m.X, np.ndarray) and not isinstance(m.X, ObsAr)
    assert isinstance(m.Y, np.ndarray) and not isinstance(m.Y, ObsAr)
    assert isinstance(m.Y_normalized, np.ndarray)


def test_set_xy_bumps_cache_version_and_accepts_ndarray():
    np.random.seed(1)
    X = np.random.randn(12, 1)
    Y = np.random.randn(12, 1)
    m = GPy.models.GPRegression(X, Y)
    v0 = m._cache_version
    kv0 = m.kern._cache_version
    m.set_XY(X=X + 0.1, Y=Y + 0.1)
    assert m._cache_version > v0
    assert m.kern._cache_version > kv0
    assert not isinstance(m.X, ObsAr)


def test_kernel_cache_invalidates_on_lengthscale_change():
    np.random.seed(2)
    X = np.random.randn(20, 1)
    Y = np.sin(X) + 0.1 * np.random.randn(20, 1)
    k = GPy.kern.RBF(1)
    m = GPy.models.GPRegression(X, Y, k)
    K1 = np.array(k.K(m.X), copy=True)
    k.lengthscale[:] = 2.5
    K2 = np.array(k.K(m.X), copy=True)
    assert not np.allclose(K1, K2)


def test_cacher_accepts_ndarray_inputs():
    class Holder:
        def __init__(self):
            self._cache_version = 0
            self.cache = FunctionCache()

        @Cache_this(limit=3, ignore_args=())
        def f(self, A):
            return A.sum()

    h = Holder()
    A = np.arange(5.0)
    assert h.f(A) == 10.0
    assert h.f(A) == 10.0  # cache hit
    h._cache_version += 1
    assert h.f(A) == 10.0  # miss then recompute


def test_obsar_module_alias_registered():
    assert "GPy.core.parameterization.observable_array" in sys.modules
    mod = sys.modules["GPy.core.parameterization.observable_array"]
    assert hasattr(mod, "ObsAr")
    assert mod.ObsAr is ObsAr


def test_pickle_roundtrip_gp_with_ndarray_xy(tmp_path):
    np.random.seed(3)
    X = np.random.randn(10, 1)
    Y = np.random.randn(10, 1)
    m = GPy.models.GPRegression(X, Y)
    path = tmp_path / "m.pickle"
    m.pickle(str(path))
    m2 = GPy.load(str(path))
    assert np.allclose(m.log_likelihood(), m2.log_likelihood())
    assert isinstance(m2.X, np.ndarray)


def test_load_previous_release_obsar_pickle_fixture():
    """Simulate a previous-release pickle that stored ObsAr under the paramz path."""
    np.random.seed(4)
    X = ObsAr(np.random.randn(8, 1))
    Y = ObsAr(np.random.randn(8, 1))
    buf = io.BytesIO()
    pickle.dump({"X": X, "Y": Y, "cls": ObsAr}, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    data = pickle.load(buf)
    # ObsAr is still the paramz class re-exported via the GPy façade (Stage 0/3).
    assert isinstance(data["X"], ObsAr)
    assert data["X"].__class__.__module__ == "paramz.core.observable_array"
    assert np.allclose(np.asarray(data["X"]), np.asarray(X))


def test_as_data_array_from_obsar():
    a = ObsAr([[1.0, 2.0], [3.0, 4.0]])
    b = _as_data_array(a)
    assert isinstance(b, np.ndarray) and not isinstance(b, ObsAr)
    assert b.shape == (2, 2)
