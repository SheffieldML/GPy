# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""CIP-000F Stage 3: version-counter cache and ndarray X/Y."""

import copy
import io
import pickle
import sys
from pickle import PickleError

import numpy as np
import pytest

import GPy
from GPy.core.gp import _as_data_array
from GPy.core.parameterization import ObsAr, Param
from GPy.core.parameterization.caching import Cache_this, Cacher, FunctionCache


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

        @Cache_this(limit=3, ignore_args=())
        def f(self, A):
            return A.sum()

    h = Holder()
    A = np.arange(5.0)
    assert h.f(A) == 10.0
    assert hasattr(h, "cache")
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
    # ObsAr stays reachable via the GPy façade; Stage 4 vendors the class under
    # ``_paramz`` while keeping the historical ``paramz.*`` sys.modules alias.
    assert isinstance(data["X"], ObsAr)
    assert "paramz.core.observable_array" in sys.modules
    assert np.allclose(np.asarray(data["X"]), np.asarray(X))


def test_as_data_array_from_obsar():
    a = ObsAr([[1.0, 2.0], [3.0, 4.0]])
    b = _as_data_array(a)
    assert isinstance(b, np.ndarray) and not isinstance(b, ObsAr)
    assert b.shape == (2, 2)


def test_sparse_gplvm_keeps_linked_param_x():
    np.random.seed(7)
    Y = np.random.randn(10, 1)
    m = GPy.models.SparseGPLVM(Y, 2, num_inducing=3)
    assert isinstance(m.X, Param)


def test_cacher_disable_enable_and_force_kwargs():
    calls = []

    def op(a, force=None):
        calls.append(force)
        return a + 1

    c = Cacher(op, limit=2, force_kwargs=("force",))
    assert c(1) == 2
    assert c(1) == 2  # hit
    assert len(calls) == 1
    assert c(1, force=True) == 2  # bypass
    assert len(calls) == 2
    c.disable_cacher()
    assert c(3) == 4
    assert len(calls) == 3
    c.enable_cacher()
    assert c(3) == 4
    assert c.__name__ == "op"
    assert "Cacher" in str(c)
    assert isinstance(copy.deepcopy(c), Cacher)
    with pytest.raises(PickleError):
        c.__getstate__()
    with pytest.raises(PickleError):
        c.__setstate__()


def test_cacher_eviction_and_function_cache_controls():
    class Holder:
        def __init__(self):
            self._cache_version = 0
            self.cache = FunctionCache()

        @Cache_this(limit=2, ignore_args=())
        def f(self, A):
            return float(A.sum())

    h = Holder()
    a, b, c = np.array([1.0]), np.array([2.0]), np.array([3.0])
    assert h.f(a) == 1.0
    assert h.f(b) == 2.0
    assert h.f(c) == 3.0  # evicts oldest
    h.cache.disable_caching()
    assert h.cache.caching_enabled is False
    h.cache.enable_caching()
    assert h.cache.caching_enabled is True
    h.cache.reset()
    assert h.f(a) == 1.0


def test_cacher_updates_when_inputs_marked_changed():
    n = [0]

    def op(a):
        n[0] += 1
        return a + n[0]

    c = Cacher(op, limit=3)
    assert c(10) == 11
    cid = c.prepare_cache_id([10])
    assert cid in c.inputs_changed
    c.inputs_changed[cid] = True
    assert c(10) == 12  # refresh existing cache entry
    A = ObsAr(np.arange(4.0))
    c2 = Cacher(lambda a: float(np.asarray(a).sum()), limit=3)
    assert c2(A) == 6.0
    c2.on_cache_changed(A)


def test_cacher_evicts_observable_and_bound_prepare_id():
    class Holder:
        def __init__(self):
            self._cache_version = 0

        def op(self, A):
            return float(np.asarray(A).sum())

    h = Holder()
    bound = h.op
    c = Cacher(bound, limit=2)
    a, b, d = ObsAr([1.0]), ObsAr([2.0]), ObsAr([3.0])
    assert c(a) == 1.0
    assert c(b) == 2.0
    assert c(d) == 3.0  # evicts ``a`` and drops its observer
    # Bound method supplies owner via operation.__self__ when owner=None
    cid = c.prepare_cache_id([a])
    assert isinstance(cid, str) and len(cid) > 0


def test_cacher_propagates_operation_errors():
    def boom(_):
        raise ValueError("nope")

    c = Cacher(boom, limit=2)
    with pytest.raises(ValueError, match="nope"):
        c(1)


def test_multioutput_gp_set_xy_stores_ndarray():
    np.random.seed(5)
    X1 = np.random.randn(8, 1)
    X2 = np.random.randn(8, 1)
    Y1 = np.sin(X1)
    Y2 = np.cos(X2)
    k = GPy.kern.RBF(1)
    m = GPy.models.MultioutputGP(
        [X1, X2],
        [Y1, Y2],
        [k, GPy.kern.DiffKern(k, 0)],
        [GPy.likelihoods.Gaussian(), GPy.likelihoods.Gaussian()],
    )
    v0 = m._cache_version
    m.set_XY([X1 + 0.1, X2 + 0.1], [Y1 + 0.1, Y2 - 0.1])
    assert isinstance(m.X, np.ndarray) and not isinstance(m.X, ObsAr)
    assert isinstance(m.Y, np.ndarray) and not isinstance(m.Y, ObsAr)
    assert m._cache_version > v0
    m.set_XY(X=[X1, X2])
    m.set_XY(Y=[Y1, Y2])
    assert isinstance(m.X, np.ndarray) and isinstance(m.Y, np.ndarray)


def test_tp_regression_set_x_y_ndarray():
    np.random.seed(6)
    X = np.random.randn(12, 1)
    Y = np.sin(X) + 0.05 * np.random.randn(12, 1)
    m = GPy.models.TPRegression(X, Y)
    m.set_X(X + 0.2)
    m.set_Y(Y + 0.1)
    assert isinstance(m.X, np.ndarray) and not isinstance(m.X, ObsAr)
    assert isinstance(m.Y, np.ndarray) and not isinstance(m.Y, ObsAr)
    m2 = GPy.models.TPRegression(X, Y, normalizer=True)
    m2.set_XY(X + 0.1, Y + 0.2)
    assert isinstance(m2.Y_normalized, np.ndarray)
