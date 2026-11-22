# Copyright (c) 2026 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Tests for SkewGaussian and SkewExponential likelihoods (CIP-0014)."""

import numpy as np
import pytest
from scipy import stats

import GPy
from GPy.models import GradientChecker

fixed_seed = 7


class TestSkewGaussianDensity:
    def test_matches_scipy_skewnorm(self):
        lik = GPy.likelihoods.SkewGaussian(variance=0.81, alpha=2.5)
        f = np.array([[0.0], [-1.0], [0.5]])
        y = np.array([[0.3], [-0.2], [1.1]])
        expected = stats.skewnorm.logpdf(y, 2.5, loc=f, scale=0.9)
        np.testing.assert_allclose(lik.logpdf(f, y), expected, rtol=1e-10)

    def test_alpha_zero_matches_gaussian(self):
        var = 0.64
        skew = GPy.likelihoods.SkewGaussian(variance=var, alpha=0.0)
        gauss = GPy.likelihoods.Gaussian(variance=var)
        f = np.linspace(-2, 2, 7)[:, None]
        y = np.linspace(-1, 1, 7)[:, None]
        np.testing.assert_allclose(skew.logpdf(f, y), gauss.logpdf(f, y), rtol=1e-10, atol=1e-12)

    def test_pdf_integrates_to_one(self):
        lik = GPy.likelihoods.SkewGaussian(variance=1.0, alpha=-1.5)
        f = 0.25
        ys = np.linspace(-8, 8, 4001)
        dens = np.exp(lik.logpdf(np.full_like(ys, f)[:, None], ys[:, None]).ravel())
        mass = np.trapezoid(dens, ys) if hasattr(np, "trapezoid") else np.trapz(dens, ys)
        np.testing.assert_allclose(mass, 1.0, atol=1e-3)


class TestSkewExponentialDensity:
    def test_alpha_zero_matches_laplace(self):
        rate = 1.7
        lik = GPy.likelihoods.SkewExponential(rate=rate, alpha=0.0)
        f = np.array([[0.0], [1.0], [-0.5]])
        y = np.array([[0.2], [0.5], [-1.0]])
        # p(e) = λ exp(-λ|e|) Φ(0) = λ exp(-λ|e|) * 0.5, and 2*(λ/2)exp = λ exp
        # When α=0: λ exp(-λ|e|) * 0.5 — wait, construction is λ exp Φ with Φ(0)=1/2
        # so density is (λ/2) exp(-λ|e|) which is NOT Laplace.
        #
        # Azzalini: 2 g G with g = (λ/2) exp → 2*(λ/2)*exp*Φ = λ exp Φ.
        # At α=0, Φ=1/2 → (λ/2) exp = Laplace. Good — matches stats.laplace.
        expected = stats.laplace.logpdf(y, loc=f, scale=1.0 / rate)
        np.testing.assert_allclose(lik.logpdf(f, y), expected, rtol=1e-10)

    def test_pdf_integrates_to_one(self):
        lik = GPy.likelihoods.SkewExponential(rate=0.8, alpha=2.0)
        f = -0.1
        ys = np.linspace(-12, 12, 5001)
        dens = np.exp(lik.logpdf(np.full_like(ys, f)[:, None], ys[:, None]).ravel())
        mass = np.trapezoid(dens, ys) if hasattr(np, "trapezoid") else np.trapz(dens, ys)
        np.testing.assert_allclose(mass, 1.0, atol=2e-3)


class TestSkewGradients:
    @pytest.mark.parametrize(
        "likelihood",
        [
            GPy.likelihoods.SkewGaussian(variance=0.7, alpha=1.2),
            GPy.likelihoods.SkewExponential(rate=1.1, alpha=-0.8),
        ],
        ids=["SkewGaussian", "SkewExponential"],
    )
    def test_dlogpdf_df(self, likelihood):
        np.random.seed(fixed_seed)
        f = np.random.randn(8, 1)
        y = f + 0.3 * np.random.randn(8, 1)
        checker = GradientChecker(
            lambda ff: likelihood.logpdf(ff, y),
            lambda ff: likelihood.dlogpdf_df(ff, y),
            f.copy(),
        )
        assert checker.checkgrad(verbose=False)

    @pytest.mark.parametrize(
        "likelihood",
        [
            GPy.likelihoods.SkewGaussian(variance=0.7, alpha=1.2),
            GPy.likelihoods.SkewExponential(rate=1.1, alpha=-0.8),
        ],
        ids=["SkewGaussian", "SkewExponential"],
    )
    def test_d2logpdf_df2(self, likelihood):
        np.random.seed(fixed_seed)
        f = np.random.randn(8, 1)
        y = f + 0.3 * np.random.randn(8, 1)
        checker = GradientChecker(
            lambda ff: likelihood.dlogpdf_df(ff, y),
            lambda ff: likelihood.d2logpdf_df2(ff, y),
            f.copy(),
        )
        assert checker.checkgrad(verbose=False)

    def test_skew_gaussian_param_gradients(self):
        np.random.seed(fixed_seed)
        lik = GPy.likelihoods.SkewGaussian(variance=0.5, alpha=0.9)
        f = np.random.randn(6, 1)
        y = f + 0.2 * np.random.randn(6, 1)

        def loglik_var(v):
            lik["variance"] = np.asarray(v).item()
            return lik.logpdf(f, y).sum()

        def dloglik_var(v):
            lik["variance"] = np.asarray(v).item()
            return lik.dlogpdf_dtheta(f, y)[0].sum()

        assert GradientChecker(loglik_var, dloglik_var, np.array([0.5])).checkgrad(verbose=False)

        def loglik_alpha(a):
            lik["alpha"] = np.asarray(a).item()
            return lik.logpdf(f, y).sum()

        def dloglik_alpha(a):
            lik["alpha"] = np.asarray(a).item()
            return lik.dlogpdf_dtheta(f, y)[1].sum()

        assert GradientChecker(loglik_alpha, dloglik_alpha, np.array([0.9])).checkgrad(verbose=False)

    def test_skew_exponential_param_gradients(self):
        np.random.seed(fixed_seed)
        lik = GPy.likelihoods.SkewExponential(rate=1.3, alpha=0.7)
        f = np.random.randn(6, 1)
        y = f + 0.2 * np.random.randn(6, 1)

        def loglik_rate(r):
            lik["rate"] = np.asarray(r).item()
            return lik.logpdf(f, y).sum()

        def dloglik_rate(r):
            lik["rate"] = np.asarray(r).item()
            return lik.dlogpdf_dtheta(f, y)[0].sum()

        assert GradientChecker(loglik_rate, dloglik_rate, np.array([1.3])).checkgrad(verbose=False)

        def loglik_alpha(a):
            lik["alpha"] = np.asarray(a).item()
            return lik.logpdf(f, y).sum()

        def dloglik_alpha(a):
            lik["alpha"] = np.asarray(a).item()
            return lik.dlogpdf_dtheta(f, y)[1].sum()

        assert GradientChecker(loglik_alpha, dloglik_alpha, np.array([0.7])).checkgrad(verbose=False)


class TestSkewInference:
    def test_skew_gaussian_laplace_smoke(self):
        np.random.seed(fixed_seed)
        X = np.linspace(0, 1, 20)[:, None]
        f_true = np.sin(2 * np.pi * X)
        Y = stats.skewnorm.rvs(2.0, loc=f_true, scale=0.2)
        model = GPy.core.GP(
            X,
            Y,
            GPy.kern.RBF(1),
            GPy.likelihoods.SkewGaussian(variance=0.1, alpha=1.0),
            inference_method=GPy.inference.latent_function_inference.Laplace(),
        )
        model.optimize(max_iters=50)
        assert np.isfinite(model.log_likelihood())
        mean, var = model.predict(X[:3])
        assert mean.shape == (3, 1) and var.shape == (3, 1)
        assert np.all(np.isfinite(mean)) and np.all(np.isfinite(var))

    def test_skew_exponential_laplace_smoke(self):
        np.random.seed(fixed_seed)
        X = np.linspace(0, 1, 20)[:, None]
        f_true = np.sin(2 * np.pi * X)
        # Approximate skew-Laplace draws via the likelihood sampler
        lik = GPy.likelihoods.SkewExponential(rate=2.0, alpha=1.5)
        Y = lik.samples(f_true)
        model = GPy.core.GP(
            X,
            Y,
            GPy.kern.RBF(1),
            GPy.likelihoods.SkewExponential(rate=1.0, alpha=1.0),
            inference_method=GPy.inference.latent_function_inference.Laplace(),
        )
        model.optimize(max_iters=50)
        assert np.isfinite(model.log_likelihood())


class TestSkewSerialization:
    def test_skew_gaussian_roundtrip(self):
        lik = GPy.likelihoods.SkewGaussian(variance=0.55, alpha=-1.3)
        restored = GPy.likelihoods.likelihood.Likelihood.from_dict(lik.to_dict())
        assert type(restored) is type(lik)
        np.testing.assert_allclose(restored.variance, lik.variance)
        np.testing.assert_allclose(restored.alpha, lik.alpha)

    def test_skew_exponential_roundtrip(self):
        lik = GPy.likelihoods.SkewExponential(rate=2.2, alpha=0.4)
        restored = GPy.likelihoods.likelihood.Likelihood.from_dict(lik.to_dict())
        assert type(restored) is type(lik)
        np.testing.assert_allclose(restored.rate, lik.rate)
        np.testing.assert_allclose(restored.alpha, lik.alpha)


class TestSkewSamples:
    def test_skew_gaussian_sample_moments(self):
        np.random.seed(fixed_seed)
        lik = GPy.likelihoods.SkewGaussian(variance=1.0, alpha=3.0)
        samples = lik.samples(np.zeros((20000, 1)))
        # Theoretical mean: δ √(2/π) with δ = α/√(1+α²)
        delta = 3.0 / np.sqrt(1.0 + 9.0)
        np.testing.assert_allclose(samples.mean(), delta * np.sqrt(2.0 / np.pi), rtol=0.05)

    def test_skew_exponential_samples_finite(self):
        np.random.seed(fixed_seed)
        lik = GPy.likelihoods.SkewExponential(rate=1.0, alpha=2.0)
        samples = lik.samples(np.array([[0.0], [1.0], [-1.0]]))
        assert samples.shape == (3, 1)
        assert np.all(np.isfinite(samples))
        # Positive skew → mean above location for f=0
        many = lik.samples(np.zeros((5000, 1)))
        assert many.mean() > 0.0
