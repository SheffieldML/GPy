# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""CIP-000F Stage 4: vendored paramz aliases and no external dependency need."""

import pickle
import sys

import numpy as np

import GPy
import paramz


def test_paramz_module_is_vendored():
    assert "GPy/core/parameterization/_paramz" in paramz.__file__.replace("\\", "/")


def test_paramz_core_aliases_registered():
    assert "paramz.core.observable_array" in sys.modules
    assert "paramz.core.observable" in sys.modules
    from paramz.core.observable_array import ObsAr

    a = ObsAr(np.array([[1.0, 2.0]]))
    assert a.shape == (1, 2)


def test_gp_pickle_roundtrip_with_vendored_paramz(tmp_path):
    np.random.seed(0)
    X = np.random.randn(12, 1)
    Y = np.sin(X)
    m = GPy.models.GPRegression(X, Y)
    path = tmp_path / "m.pkl"
    m.pickle(str(path))
    m2 = GPy.load(str(path))
    assert np.allclose(m.log_likelihood(), m2.log_likelihood())


def test_legacy_paramz_path_pickle_loads():
    """Objects pickled under paramz.* names unpickle via aliases."""
    from paramz.core.observable_array import ObsAr

    blob = pickle.dumps(ObsAr(np.arange(4.0).reshape(2, 2)))
    obj = pickle.loads(blob)
    assert np.allclose(np.asarray(obj), np.arange(4.0).reshape(2, 2))
