"""The Kuu jitter of the sparse inference methods is a constructor option."""

import numpy as np
import pytest
import GPy
from GPy.inference.latent_function_inference import FITC, VarDTC


def _model(inf):
    rng = np.random.RandomState(1)
    X = rng.rand(40, 1)
    Y = np.sin(6 * X) + 0.05 * rng.randn(40, 1)
    m = GPy.models.SparseGPRegression(X, Y, GPy.kern.RBF(1, lengthscale=0.8), Z=np.linspace(0, 1, 12)[:, None])
    m.inference_method = inf
    m.parameters_changed()  # recompute with the new inference method
    return m


@pytest.mark.parametrize("cls,default", [(VarDTC, 1e-8), (FITC, 1e-6)])
def test_defaults_are_the_historical_values(cls, default):
    assert cls().const_jitter == default
    np.testing.assert_allclose(
        _model(cls()).log_likelihood(), _model(cls(const_jitter=default)).log_likelihood(), rtol=1e-14
    )


@pytest.mark.parametrize("cls", [VarDTC, FITC])
def test_jitter_is_used(cls):
    base = _model(cls()).log_likelihood()
    big = _model(cls(const_jitter=1e-2)).log_likelihood()
    assert abs(big - base) > 1e-6


def test_vardtc_matching_jitter_gives_matching_values():
    # an ill conditioned Kuu: nearby inducing inputs make jitter matter
    m1, m2 = _model(VarDTC(const_jitter=1e-6)), _model(VarDTC())
    m1.Z[:] = m2.Z[:] = np.linspace(0, 0.05, 12)[:, None]
    assert abs(m1.log_likelihood() - m2.log_likelihood()) > 0
    m3 = _model(VarDTC(const_jitter=1e-6))
    m3.Z[:] = m1.Z[:]
    np.testing.assert_allclose(m1.log_likelihood(), m3.log_likelihood(), rtol=1e-14)


def test_vardtc_serialization_records_the_jitter():
    assert VarDTC().to_dict()["const_jitter"] == 1e-8
    d = VarDTC(const_jitter=1e-6).to_dict()
    assert VarDTC.from_dict(d).const_jitter == 1e-6
