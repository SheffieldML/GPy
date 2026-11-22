# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""CIP-0013: sparse_linalg tokens, LQ factor, VarDTC/FITC mode parity."""

import numpy as np
import pytest

import GPy
from GPy.util.sparse_linalg import (
    parse_sparse_linalg,
    lq_factor,
    lq_adjoint,
    factor_I_plus_AA,
)


def test_parse_sparse_linalg_canonical_and_aliases():
    assert parse_sparse_linalg("lq") == "lq"
    assert parse_sparse_linalg("STABLE") == "stable"
    assert parse_sparse_linalg("chol") == "stable"
    assert parse_sparse_linalg("StableFactor") == "stable"
    assert parse_sparse_linalg("gram") == "legacy"
    with pytest.raises(ValueError, match="sparse_linalg"):
        parse_sparse_linalg("bogus")
    with pytest.raises(TypeError, match="sparse_linalg must be a string"):
        parse_sparse_linalg(1)  # type: ignore[arg-type]


def test_lq_factor_rejects_tall_matrix():
    with pytest.raises(ValueError, match="wide matrix"):
        lq_factor(np.eye(5, 3))


def test_lq_factor_matches_gram_chol():
    rng = np.random.default_rng(0)
    m, n = 5, 40
    F = rng.normal(size=(m, n)) / np.sqrt(n)
    L, Q = lq_factor(np.hstack([np.eye(m), F]))
    B = np.eye(m) + F @ F.T
    Lc = np.linalg.cholesky(B)
    assert np.allclose(L @ L.T, B, atol=1e-12)
    assert np.allclose(L @ Q, np.hstack([np.eye(m), F]), atol=1e-12)
    assert np.allclose(np.abs(np.diag(L)), np.abs(np.diag(Lc)), atol=1e-12)


def test_lq_adjoint_finite_difference():
    rng = np.random.default_rng(1)
    m, n = 4, 12
    W = rng.normal(size=(m, n))
    # Skip sign flips so the (L, Q) map is locally smooth for FD.
    L, Q = lq_factor(W, normalize_signs=False)

    def phi(L_, Q_):
        return float(np.sum(L_) + np.sum(Q_))

    Lbar = np.ones_like(L)
    Qbar = np.ones_like(Q)
    Wbar = lq_adjoint(L, Q, Lbar, Qbar)
    eps = 1e-7
    num = np.zeros_like(W)
    for i in range(m):
        for j in range(n):
            Wp = W.copy()
            Wm = W.copy()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            num[i, j] = (
                phi(*lq_factor(Wp, normalize_signs=False))
                - phi(*lq_factor(Wm, normalize_signs=False))
            ) / (2 * eps)
    assert np.allclose(Wbar, num, rtol=1e-4, atol=1e-5)


def test_factor_I_plus_AA_legacy_raises():
    F = np.eye(3, 5)
    with pytest.raises(NotImplementedError, match="legacy"):
        factor_I_plus_AA(F, "legacy")


def test_vardtc_lq_matches_stable_ll():
    rng = np.random.default_rng(2)
    N, M, Q = 40, 8, 2
    X = rng.uniform(-2, 2, size=(N, Q))
    Y = np.sin(X[:, :1]) + 0.1 * rng.normal(size=(N, 1))
    Z = X[:M].copy()
    k = GPy.kern.RBF(Q, lengthscale=1.0) + GPy.kern.White(Q, variance=1e-4)
    m_s = GPy.models.SparseGPRegression(X, Y, kernel=k.copy(), Z=Z.copy())
    m_l = GPy.models.SparseGPRegression(X, Y, kernel=k.copy(), Z=Z.copy())
    m_s.inference_method = GPy.inference.latent_function_inference.VarDTC(sparse_linalg="stable")
    m_l.inference_method = GPy.inference.latent_function_inference.VarDTC(sparse_linalg="lq")
    assert m_s.inference_method.sparse_linalg == "stable"
    assert m_l.inference_method.sparse_linalg == "lq"
    assert m_l.log_likelihood() == pytest.approx(m_s.log_likelihood(), rel=1e-10, abs=1e-8)


def test_fitc_lq_matches_stable_ll():
    rng = np.random.default_rng(3)
    N, M, Q = 40, 8, 2
    X = rng.uniform(-2, 2, size=(N, Q))
    Y = np.sin(X[:, :1]) + 0.1 * rng.normal(size=(N, 1))
    Z = X[:M].copy()
    k = GPy.kern.RBF(Q, lengthscale=1.0)
    m_s = GPy.models.SparseGPRegression(X, Y, kernel=k.copy(), Z=Z.copy())
    m_l = GPy.models.SparseGPRegression(X, Y, kernel=k.copy(), Z=Z.copy())
    m_s.inference_method = GPy.inference.latent_function_inference.FITC(sparse_linalg="stable")
    m_l.inference_method = GPy.inference.latent_function_inference.FITC(sparse_linalg="lq")
    assert m_l.log_likelihood() == pytest.approx(m_s.log_likelihood(), rel=1e-10, abs=1e-8)


def test_sparse_linalg_property_aliases():
    inf = GPy.inference.latent_function_inference.VarDTC()
    assert inf.sparse_linalg == "stable"
    inf.sparse_linalg = "LQ"
    assert inf.sparse_linalg == "lq"
    inf.sparse_linalg = "chol"
    assert inf.sparse_linalg == "stable"

    fitc = GPy.inference.latent_function_inference.FITC()
    assert fitc.sparse_linalg == "stable"
    fitc.sparse_linalg = "LQ"
    assert fitc.sparse_linalg == "lq"
    fitc.sparse_linalg = "chol"
    assert fitc.sparse_linalg == "stable"


def test_vardtc_to_dict_and_pickle_roundtrip():
    import pickle

    inf = GPy.inference.latent_function_inference.VarDTC(
        limit=2, sparse_linalg="lq"
    )
    d = inf.to_dict()
    assert d["sparse_linalg"] == "lq"
    assert d["limit"] == 2

    restored = pickle.loads(pickle.dumps(inf))
    assert restored.sparse_linalg == "lq"
    assert restored.limit == 2

    # Older pickles stored only ``limit``.
    legacy = GPy.inference.latent_function_inference.VarDTC()
    legacy.__setstate__(3)
    assert legacy.limit == 3
    assert legacy.sparse_linalg == "stable"


def test_vardtc_lq_uncertain_inputs_falls_back_to_stable(caplog):
    import logging

    rng = np.random.default_rng(4)
    N, M, Q = 30, 6, 2
    X = rng.uniform(-1, 1, size=(N, Q))
    Y = np.sin(X[:, :1]) + 0.05 * rng.normal(size=(N, 1))
    Z = X[:M].copy()
    X_var = 0.05 * rng.random(size=X.shape) + 1e-3
    k = GPy.kern.RBF(Q, lengthscale=1.0)
    m = GPy.models.SparseGPRegression(
        X, Y, kernel=k, Z=Z, X_variance=X_var
    )
    inf = GPy.inference.latent_function_inference.VarDTC(sparse_linalg="lq")
    with caplog.at_level(logging.INFO, logger="vardtc"):
        _post, log_marginal, _grad = inf.inference(
            m.kern, m.X, m.Z, m.likelihood, m.Y
        )
    assert np.isfinite(np.asarray(log_marginal)).all()
    assert any("uncertain inputs" in r.getMessage() for r in caplog.records)
