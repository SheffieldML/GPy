# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Tests for optional chunked / threaded RBF and linear psi-statistics (CIP-0010)."""

from unittest.mock import patch

import numpy as np
import pytest

from GPy.core.parameterization.variational import NormalPosterior, SpikeAndSlabPosterior
from GPy.kern import RBF, Linear
from GPy.kern.src.psi_comp import PSICOMP_RBF, PSICOMP_Linear, rbf_psi_comp, linear_psi_comp
from GPy.models import BayesianGPLVM


def _vp(N=40, Q=3, seed=0):
    rng = np.random.RandomState(seed)
    mu = rng.randn(N, Q)
    S = rng.rand(N, Q) + 0.05
    return NormalPosterior(mu, S), rng


@pytest.mark.parametrize("chunk_size", [7, 10, 13, 40, None])
@pytest.mark.parametrize("n_threads", [1, 2])
@pytest.mark.parametrize("return_psi2_n", [False, True])
def test_rbf_chunked_matches_unchunked(chunk_size, n_threads, return_psi2_n):
    vp, rng = _vp()
    Z = rng.randn(8, 3)
    variance = np.array([1.2])
    lengthscale = np.array([0.7, 1.1, 0.9])

    ref = rbf_psi_comp.psicomputations(variance, lengthscale, Z, vp, return_psi2_n=return_psi2_n)
    got = rbf_psi_comp.psicomputations(
        variance,
        lengthscale,
        Z,
        vp,
        return_psi2_n=return_psi2_n,
        chunk_size=chunk_size,
        n_threads=n_threads,
    )
    for a, b in zip(ref, got, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    d0 = rng.randn(vp.mean.shape[0])
    d1 = rng.randn(vp.mean.shape[0], Z.shape[0])
    if return_psi2_n:
        d2 = rng.randn(vp.mean.shape[0], Z.shape[0], Z.shape[0])
        d2 = d2 + np.swapaxes(d2, 1, 2)
    else:
        d2 = rng.randn(Z.shape[0], Z.shape[0])
        d2 = d2 + d2.T

    ref_g = rbf_psi_comp.psiDerivativecomputations(d0, d1, d2, variance, lengthscale, Z, vp)
    got_g = rbf_psi_comp.psiDerivativecomputations(
        d0, d1, d2, variance, lengthscale, Z, vp, chunk_size=chunk_size, n_threads=n_threads
    )
    for a, b in zip(ref_g, got_g, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("chunk_size", [7, 13, None])
@pytest.mark.parametrize("n_threads", [1, 2])
@pytest.mark.parametrize("return_psi2_n", [False, True])
def test_linear_chunked_matches_unchunked(chunk_size, n_threads, return_psi2_n):
    vp, rng = _vp()
    Z = rng.randn(8, 3)
    variance = np.array([0.8, 1.1, 0.5])

    ref = linear_psi_comp.psicomputations(variance, Z, vp, return_psi2_n=return_psi2_n)
    got = linear_psi_comp.psicomputations(
        variance, Z, vp, return_psi2_n=return_psi2_n, chunk_size=chunk_size, n_threads=n_threads
    )
    for a, b in zip(ref, got, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    d0 = rng.randn(vp.mean.shape[0])
    d1 = rng.randn(vp.mean.shape[0], Z.shape[0])
    if return_psi2_n:
        d2 = rng.randn(vp.mean.shape[0], Z.shape[0], Z.shape[0])
        d2 = d2 + np.swapaxes(d2, 1, 2)
    else:
        d2 = rng.randn(Z.shape[0], Z.shape[0])
        d2 = d2 + d2.T

    ref_g = linear_psi_comp.psiDerivativecomputations(d0, d1, d2, variance, Z, vp)
    got_g = linear_psi_comp.psiDerivativecomputations(
        d0, d1, d2, variance, Z, vp, chunk_size=chunk_size, n_threads=n_threads
    )
    for a, b in zip(ref_g, got_g, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_rbf_isotropic_chunked_derivatives():
    """Non-ARD lengthscale path (``len(lengthscale) == 1``) under chunking."""
    vp, rng = _vp(N=20, Q=2)
    Z = rng.randn(5, 2)
    variance = np.array([1.1])
    lengthscale = np.array([0.8])
    d0 = rng.randn(20)
    d1 = rng.randn(20, 5)
    d2 = rng.randn(5, 5)
    d2 = d2 + d2.T
    ref = rbf_psi_comp.psiDerivativecomputations(d0, d1, d2, variance, lengthscale, Z, vp)
    got = rbf_psi_comp.psiDerivativecomputations(d0, d1, d2, variance, lengthscale, Z, vp, chunk_size=6, n_threads=2)
    for a, b in zip(ref, got, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
    assert np.ndim(got[1]) == 0 or np.asarray(got[1]).shape == ()


def test_kernel_psicomp_chunk_size_and_checkgrad():
    rng = np.random.RandomState(1)
    N, Q, M = 25, 2, 5
    Y = rng.randn(N, 4)
    k = RBF(Q, ARD=True)
    k.psicomp.chunk_size = 8
    k.psicomp.n_threads = 2
    m = BayesianGPLVM(Y, Q, kernel=k, num_inducing=M)
    assert m.checkgrad()


def test_linear_kernel_psi_expectations_match_with_chunking():
    vp, rng = _vp(N=30, Q=2)
    Z = rng.randn(6, 2)
    k = Linear(2, ARD=True)
    k.randomize()
    ref0, ref1, ref2 = k.psi0(Z, vp), k.psi1(Z, vp), k.psi2(Z, vp)
    k.psicomp.chunk_size = 9
    k.psicomp.n_threads = 2
    got0, got1, got2 = k.psi0(Z, vp), k.psi1(Z, vp), k.psi2(Z, vp)
    np.testing.assert_allclose(ref0, got0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref1, got1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref2, got2, rtol=1e-10, atol=1e-10)

    w0 = rng.randn(30)
    w1 = rng.randn(30, Z.shape[0])
    w2 = rng.randn(Z.shape[0], Z.shape[0])
    w2 = w2 + w2.T
    k.psicomp.chunk_size = None
    k.update_gradients_expectations(w0, w1, w2, Z, vp)
    g_ref = k.gradient.copy()
    k.psicomp.chunk_size = 9
    k.update_gradients_expectations(w0, w1, w2, Z, vp)
    np.testing.assert_allclose(g_ref, k.gradient, rtol=1e-9, atol=1e-9)


def test_kernel_psi_expectations_match_with_chunking():
    vp, rng = _vp(N=30, Q=2)
    Z = rng.randn(6, 2)
    k = RBF(2, ARD=True)
    k.randomize()
    ref0, ref1, ref2 = k.psi0(Z, vp), k.psi1(Z, vp), k.psi2(Z, vp)
    k.psicomp.chunk_size = 9
    k.psicomp.n_threads = 2
    got0, got1, got2 = k.psi0(Z, vp), k.psi1(Z, vp), k.psi2(Z, vp)
    np.testing.assert_allclose(ref0, got0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref1, got1, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ref2, got2, rtol=1e-10, atol=1e-10)

    w0 = rng.randn(30)
    w1 = rng.randn(30, Z.shape[0])
    w2 = rng.randn(Z.shape[0], Z.shape[0])
    w2 = w2 + w2.T

    k.psicomp.chunk_size = None
    k.update_gradients_expectations(w0, w1, w2, Z, vp)
    g_ref = k.gradient.copy()
    k.psicomp.chunk_size = 9
    k.psicomp.n_threads = 2
    k.update_gradients_expectations(w0, w1, w2, Z, vp)
    np.testing.assert_allclose(g_ref, k.gradient, rtol=1e-9, atol=1e-9)


def test_psicomp_unknown_posterior_raises():
    vp, rng = _vp(N=5, Q=2)
    Z = rng.randn(3, 2)
    k_rbf = RBF(2, ARD=True)
    k_lin = Linear(2, ARD=True)
    for psicomp, kern in ((PSICOMP_RBF(), k_rbf), (PSICOMP_Linear(), k_lin)):
        with pytest.raises(ValueError, match="unknown"):
            psicomp._psicomp_compute(kern, Z, object())
        with pytest.raises(ValueError, match="unknown"):
            psicomp._psiDer_compute(kern, np.zeros(5), np.zeros((5, 3)), np.zeros((3, 3)), Z, object())


def test_psicomp_spike_and_slab_dispatch():
    """Spike-and-slab still routes to the dedicated psi modules when chunking is set."""
    rng = np.random.RandomState(2)
    N, Q, M = 6, 2, 3
    mu = rng.randn(N, Q)
    S = rng.rand(N, Q) + 0.05
    gamma = rng.rand(N, Q) * 0.5 + 0.25
    ss = SpikeAndSlabPosterior(mu, S, gamma)
    Z = rng.randn(M, Q)
    k_rbf = RBF(Q, ARD=True)
    k_lin = Linear(Q, ARD=True)
    sentinel = (np.ones(N), np.ones((N, M)), np.ones((M, M)))
    der_sentinel = (1.0, np.ones(M), np.ones((N, Q)), np.ones((N, Q)))

    with patch("GPy.kern.src.psi_comp.ssrbf_psi_comp.psicomputations", return_value=sentinel) as fwd:
        out = PSICOMP_RBF(chunk_size=3)._psicomp_compute(k_rbf, Z, ss)
        assert fwd.called
        assert out is sentinel
    with patch("GPy.kern.src.psi_comp.ssrbf_psi_comp.psiDerivativecomputations", return_value=der_sentinel) as der:
        out = PSICOMP_RBF(chunk_size=3)._psiDer_compute(k_rbf, np.zeros(N), np.zeros((N, M)), np.zeros((M, M)), Z, ss)
        assert der.called
        assert out is der_sentinel

    lin_der = (np.ones(Q), np.ones((M, Q)), np.ones((N, Q)), np.ones((N, Q)))
    with patch("GPy.kern.src.psi_comp.sslinear_psi_comp.psicomputations", return_value=sentinel) as fwd:
        out = PSICOMP_Linear(chunk_size=3)._psicomp_compute(k_lin, Z, ss)
        assert fwd.called
        assert out is sentinel
    with patch("GPy.kern.src.psi_comp.sslinear_psi_comp.psiDerivativecomputations", return_value=lin_der) as der:
        out = PSICOMP_Linear(chunk_size=3)._psiDer_compute(
            k_lin, np.zeros(N), np.zeros((N, M)), np.zeros((M, M)), Z, ss
        )
        assert der.called
        assert out is lin_der
