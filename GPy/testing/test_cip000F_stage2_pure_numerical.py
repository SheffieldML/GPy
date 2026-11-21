"""CIP-000F Stage 2 acceptance: pure LML, likelihood, and stationary cov ops."""

from __future__ import annotations

import numpy as np

import GPy
from GPy.inference.latent_function_inference import exact_gp_ops
from GPy.kern.src import stationary_cov_ops
from GPy.likelihoods import gaussian_ops


def _rng(seed=0):
    return np.random.RandomState(seed)


class TestExactGPLML:
    def test_pure_lml_without_model_or_param(self):
        rng = _rng()
        X = rng.randn(15, 1)
        Y = np.sin(X) + 0.1 * rng.randn(15, 1)
        kv, ls, noise = 1.2, 0.8, 0.05
        lml, extras = exact_gp_ops.rbf_log_marginal(X, Y, kv, ls, noise)
        assert np.isfinite(lml)
        assert extras["K"].shape == (15, 15)
        assert extras["dL_dK"].shape == (15, 15)

    def test_lml_and_dL_dK_match_gpy_model(self):
        rng = _rng(1)
        X = rng.randn(12, 2)
        Y = rng.randn(12, 1)
        m = GPy.models.GPRegression(X, Y, kernel=GPy.kern.RBF(2, variance=0.9, lengthscale=1.1))
        m.Gaussian_noise.variance = 0.07
        lml_model = float(m.log_likelihood())
        variance = float(np.asarray(m.kern.variance).reshape(-1)[0])
        lengthscale = float(np.asarray(m.kern.lengthscale).reshape(-1)[0])
        noise = float(np.asarray(m.Gaussian_noise.variance).reshape(-1)[0])
        Y_train = np.asarray(m.Y_normalized, dtype=float)
        lml_pure, extras = exact_gp_ops.rbf_log_marginal(X, Y_train, variance, lengthscale, noise)
        np.testing.assert_allclose(lml_pure, lml_model, rtol=1e-10, atol=1e-10)

        _post, log_marginal, grad_dict = m.inference_method.inference(
            m.kern, m.X, m.likelihood, m.Y_normalized, m.mean_function, m.Y_metadata
        )
        np.testing.assert_allclose(log_marginal, lml_model, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(extras["dL_dK"], grad_dict["dL_dK"], rtol=1e-9, atol=1e-9)


class TestGaussianLikelihoodOps:
    def test_logpdf_link_parity(self):
        rng = _rng(2)
        f = rng.randn(20, 1)
        y = f + 0.2 * rng.randn(20, 1)
        variance = 0.25
        lik = GPy.likelihoods.Gaussian(variance=variance)
        np.testing.assert_allclose(
            gaussian_ops.logpdf_link(f, y, variance),
            lik.logpdf_link(f, y),
        )
        np.testing.assert_allclose(
            gaussian_ops.dlogpdf_dlink(f, y, variance),
            lik.dlogpdf_dlink(f, y),
        )


class TestStationaryCovOps:
    def test_exponential_matern_parity(self):
        rng = _rng(3)
        X = rng.randn(9, 1)
        X2 = rng.randn(6, 1)
        for name, ctor, k_fn, dk_fn in (
            ("exp", GPy.kern.Exponential, stationary_cov_ops.exponential_k_of_r, stationary_cov_ops.exponential_dk_dr),
            ("m32", GPy.kern.Matern32, stationary_cov_ops.matern32_k_of_r, stationary_cov_ops.matern32_dk_dr),
            ("m52", GPy.kern.Matern52, stationary_cov_ops.matern52_k_of_r, stationary_cov_ops.matern52_dk_dr),
        ):
            kern = ctor(1, variance=1.3, lengthscale=0.7)
            r = kern._scaled_dist(X, X2)
            v = np.asarray(kern.variance).reshape(-1)[0]
            np.testing.assert_allclose(k_fn(r, v), kern.K_of_r(r), err_msg=name)
            np.testing.assert_allclose(dk_fn(r, v), kern.dK_dr(r), err_msg=name)

    def test_full_K_helpers(self):
        rng = _rng(4)
        X = rng.randn(8, 2)
        ls = np.array([0.9, 1.2])
        K_e = stationary_cov_ops.K_exponential(X, 1.0, ls, ARD=True)
        K_m = stationary_cov_ops.K_matern32(X, 1.0, ls, ARD=True)
        K_5 = stationary_cov_ops.K_matern52(X, 1.0, ls, ARD=True)
        assert K_e.shape == K_m.shape == K_5.shape == (8, 8)
        np.testing.assert_allclose(np.diag(K_e), 1.0)
        np.testing.assert_allclose(np.diag(K_m), 1.0)
        np.testing.assert_allclose(np.diag(K_5), 1.0)
