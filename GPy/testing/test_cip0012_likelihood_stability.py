"""Unit-suite coverage for CIP-0012 likelihood numerical fixes.

Codecov runs ``pytest GPy/testing --cov=GPy`` only; the broader probes in
``numerical_stability/`` do not count toward patch coverage. These tests
exercise the rewritten Bernoulli / Poisson / Gamma paths so the PR patch
stays green.
"""

import numpy as np

import GPy
from GPy.likelihoods import link_functions


class TestBernoulliCIP0012:
    def test_probit_logpdf_and_f_derivatives_finite(self):
        lik = GPy.likelihoods.Bernoulli()
        f = np.array([[-8.0], [0.0], [8.0]])
        for y in (0.0, 1.0):
            Y = np.full_like(f, y)
            assert np.all(np.isfinite(lik.logpdf(f, Y)))
            assert np.all(np.isfinite(lik.dlogpdf_df(f, Y)))
            assert np.all(np.isfinite(lik.d2logpdf_df2(f, Y)))
            assert np.all(np.isfinite(lik.d3logpdf_df3(f, Y)))

    def test_non_probit_logpdf_falls_back_to_link_space(self):
        lik = GPy.likelihoods.Bernoulli(gp_link=link_functions.Identity())
        f = np.array([[0.2], [0.8]])
        y = np.array([[1.0], [0.0]])
        assert np.all(np.isfinite(lik.logpdf(f, y)))
        assert np.all(np.isfinite(lik.dlogpdf_df(f, y)))
        assert np.all(np.isfinite(lik.d2logpdf_df2(f, y)))
        assert np.all(np.isfinite(lik.d3logpdf_df3(f, y)))

    def test_link_space_derivatives_stable_near_bounds(self):
        lik = GPy.likelihoods.Bernoulli()
        # Stay inside (0,1) enough that 1/p**2 stays finite; still exercises clipping.
        p = np.array([[1e-12], [0.5], [1.0 - 1e-12]])
        y = np.array([[1.0], [0.0], [1.0]])
        assert np.all(np.isfinite(lik.logpdf_link(p, y)))
        assert np.all(np.isfinite(lik.dlogpdf_dlink(p, y)))
        assert np.all(np.isfinite(lik.d2logpdf_dlink2(p, y)))
        assert np.all(np.isfinite(lik.d3logpdf_dlink3(p, y)))

    def test_ep_moment_match_clips_nonpositive_variance(self):
        lik = GPy.likelihoods.Bernoulli()
        # Extreme cavity: tilted variance formula can go non-positive / exceed 1/tau
        # without the CIP-0012 guard.
        for y, tau, scale in ((0.0, 1e-8, 1e5), (1.0, 1e-8, -1e5), (0.0, 1e8, 1e-3)):
            v = scale * np.sqrt(tau**2 + tau)
            Z, mu, s2 = lik.moments_match_ep(np.array(y), tau, v)
            assert np.isfinite(mu) and np.isfinite(s2)
            assert 0.0 < s2 <= (1.0 / tau) * (1.0 + 1e-9)
            assert 0.0 <= Z <= 1.0


class TestPoissonCIP0012:
    def test_log_link_latent_space_paths(self):
        lik = GPy.likelihoods.Poisson()
        f = np.array([[-800.0], [0.0], [2.0]])
        y = np.array([[0.0], [1.0], [5.0]])
        assert np.all(np.isfinite(lik.logpdf(f, y)))
        assert np.all(np.isfinite(lik.dlogpdf_df(f, y)))
        assert np.all(np.isfinite(lik.d2logpdf_df2(f, y)))
        assert np.all(np.isfinite(lik.d3logpdf_df3(f, y)))

    def test_non_log_link_falls_back(self):
        lik = GPy.likelihoods.Poisson(gp_link=link_functions.Identity())
        f = np.array([[0.5], [2.0]])
        y = np.array([[0.0], [3.0]])
        assert np.all(np.isfinite(lik.logpdf(f, y)))
        assert np.all(np.isfinite(lik.dlogpdf_df(f, y)))
        assert np.all(np.isfinite(lik.d2logpdf_df2(f, y)))
        assert np.all(np.isfinite(lik.d3logpdf_df3(f, y)))

    def test_link_space_zero_count_at_zero_rate(self):
        lik = GPy.likelihoods.Poisson()
        link_f = np.array([[0.0], [1e-300], [2.0]])
        y = np.array([[0.0], [0.0], [4.0]])
        assert np.all(np.isfinite(lik.logpdf_link(link_f, y)))
        d = lik.dlogpdf_dlink(link_f, y)
        assert np.all(np.isfinite(d))
        np.testing.assert_allclose(d[0], -1.0)


class TestGammaCIP0012:
    def test_pdf_link_via_logpdf(self):
        lik = GPy.likelihoods.Gamma(beta=1.5)
        link_f = np.array([[0.5], [1.0], [2.0]])
        y = np.array([[0.3], [1.0], [5.0]])
        pdf = lik.pdf_link(link_f, y)
        assert np.isfinite(pdf)
        assert pdf > 0.0
        np.testing.assert_allclose(pdf, np.exp(np.sum(lik.logpdf_link(link_f, y))), rtol=1e-12)
