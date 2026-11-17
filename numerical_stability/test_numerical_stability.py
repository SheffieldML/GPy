"""
Numerical stability tests for GPy.

The suite probes the places where floating point arithmetic is most likely to
go wrong in a GP library, and checks one of three kinds of contract:

* **finiteness** - extreme but representable inputs must not produce NaN/inf
  where the mathematical answer is finite;
* **accuracy** - results agree with a trusted reference (scipy, slogdet, an
  analytic identity) to a tolerance justified by conditioning;
* **invariance** - transformations that leave the maths unchanged (input
  translation/rescaling, output standardisation) leave the answer unchanged.

Layout (one section per layer, lowest first):

1. Linear algebra primitives      (GPy.util.linalg)
2. Elementwise / special functions (GPy.util.misc, GPy.util.univariate_Gaussian)
3. Link functions                  (GPy.likelihoods.link_functions)
4. Likelihoods                     (logpdf + derivatives, EP moment matching)
5. Kernels                         (PSD, symmetry, r=0 gradients, offsets)
6. Models and inference            (exact, sparse, SVGP, EP, Laplace)
7. Source hygiene                  (no live debugger breakpoints on error paths)

Known defects are recorded as ``xfail(strict=True)`` so that the suite stays
green today but fails loudly - prompting removal of the marker - as soon as a
fix lands.  Each reason string names the cause.
"""

import ast
import logging
import pathlib
import sys
import types

import numpy as np
import pytest
import scipy.special as sps
import scipy.stats as stats
from scipy import linalg as sla

import GPy
from GPy.likelihoods import link_functions as lf
from GPy.util import univariate_Gaussian as ug
from GPy.util.linalg import (
    backsub_both_sides,
    dpotrs,
    dtrtrs,
    jitchol,
    pddet,
    pdinv,
    symmetrify,
    tdot,
)
from GPy.util.misc import chain_2, chain_3, safe_cube, safe_exp, safe_quad, safe_square, safe_three_times

EPS = np.finfo(np.float64).eps
LaplaceInf = GPy.inference.latent_function_inference.Laplace


pytestmark = pytest.mark.numerical


@pytest.fixture(autouse=True)
def _quiet_numpy_and_jitter_logging():
    """Underflow warnings and jitter log lines are expected noise here."""
    logging.disable(logging.WARNING)
    with np.errstate(all="ignore"):
        yield
    logging.disable(logging.NOTSET)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _spd_with_condition(n, cond, seed=0):
    """Random SPD matrix with prescribed 2-norm condition number."""
    q, _ = np.linalg.qr(_rng(seed).standard_normal((n, n)))
    eig = np.logspace(0, -np.log10(cond), n)
    return (q * eig) @ q.T


def _regression_data(n=40, seed=1):
    rng = _rng(seed)
    X = rng.uniform(0, 10, (n, 1))
    Y = np.sin(X) + 0.1 * rng.standard_normal(X.shape)
    return X, Y


def _all_finite(model):
    return bool(np.isfinite(model.log_likelihood()) and np.all(np.isfinite(model.gradient)))


# =============================================================================
# 1. Linear algebra primitives
# =============================================================================


class TestJitchol:
    def test_well_conditioned_needs_no_jitter(self):
        A = _spd_with_condition(20, 1e3)
        L = jitchol(A)
        np.testing.assert_allclose(L @ L.T, A, rtol=0, atol=1e-13)
        assert np.allclose(L, np.tril(L))

    @pytest.mark.parametrize("scale", [1e-200, 1e-12, 1.0, 1e12, 1e200])
    def test_rank_deficient_is_recovered_scale_invariantly(self, scale):
        """Jitter is relative to mean(diag), so behaviour must not depend on units."""
        A = np.ones((6, 6)) * scale  # rank one
        L = jitchol(A)
        assert np.all(np.isfinite(L))
        rel_err = np.abs(L @ L.T - A).max() / scale
        assert rel_err < 1e-4

    def test_zero_matrix_raises(self):
        with pytest.raises(np.linalg.LinAlgError):
            jitchol(np.zeros((3, 3)))

    def test_negative_diagonal_raises(self):
        with pytest.raises(np.linalg.LinAlgError):
            jitchol(np.diag([1.0, -1.0, 1.0]))

    def test_indefinite_matrix_raises_rather_than_looping(self):
        A = np.diag([1.0, 1.0, -1e6])
        A[2, 2] = 1.0
        A[0, 2] = A[2, 0] = 10.0  # strongly indefinite, positive diagonal
        with pytest.raises(np.linalg.LinAlgError):
            jitchol(A, maxtries=5)

    @pytest.mark.parametrize(
        "where",
        [
            "nan_offdiag",
            "nan_diag",
            pytest.param(
                "inf_diag",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="jitchol trusts LAPACK info==0 and returns a factor containing inf "
                    "for +inf on the diagonal instead of raising; non-finite kernels then "
                    "surface as a NaN likelihood far from the cause",
                ),
            ),
        ],
    )
    def test_nonfinite_input_raises(self, where):
        A = np.eye(3)
        if where == "nan_offdiag":
            A[1, 0] = A[0, 1] = np.nan
        elif where == "nan_diag":
            A[1, 1] = np.nan
        else:
            A[1, 1] = np.inf
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            L = jitchol(A)
            assert np.all(np.isfinite(L))  # pragma: no cover


class TestPdinvAndSolves:
    @pytest.mark.parametrize("cond", [1e2, 1e6, 1e10])
    def test_pdinv_residual_tracks_condition_number(self, cond):
        n = 30
        A = _spd_with_condition(n, cond)
        Ai, L, Li, logdet = pdinv(A)
        resid = np.linalg.norm(Ai @ A - np.eye(n), 2)
        assert resid < 100 * n * cond * EPS
        np.testing.assert_allclose(logdet, np.linalg.slogdet(A)[1], rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(Ai, Ai.T)  # symmetrify must make it exactly symmetric
        np.testing.assert_allclose(Li @ L, np.eye(n), atol=100 * n * np.sqrt(cond) * EPS)

    def test_pdinv_hilbert(self):
        """Hilbert(10) has cond ~1.6e13 - near the edge of what float64 can invert."""
        H = sla.hilbert(10)
        Ai, _, _, logdet = pdinv(H)
        assert np.all(np.isfinite(Ai))
        np.testing.assert_allclose(logdet, np.linalg.slogdet(H)[1], rtol=1e-6)

    @pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e200, 1e300])
    def test_logdet_at_extreme_scales(self, scale):
        n = 3
        Ai, _, _, logdet = pdinv(np.eye(n) * scale)
        np.testing.assert_allclose(logdet, n * np.log(scale), rtol=1e-12)
        assert np.isfinite(pddet(np.eye(n) * scale))

    def test_dtrtrs_and_dpotrs_residuals(self):
        n = 25
        A = _spd_with_condition(n, 1e8, seed=3)
        B = _rng(4).standard_normal((n, 4))
        L = jitchol(A)
        X, info = dtrtrs(L, B, lower=1)
        assert info == 0
        np.testing.assert_allclose(L @ X, B, atol=1e-10)
        X2, info = dpotrs(L, B, lower=1)
        assert info == 0
        rel = np.linalg.norm(A @ X2 - B) / (np.linalg.norm(A) * np.linalg.norm(X2))
        assert rel < 1e3 * EPS

    def test_tdot_is_exactly_symmetric(self):
        M = _rng(5).standard_normal((50, 7)) * 1e150
        T = tdot(M)
        np.testing.assert_array_equal(T, T.T)
        assert np.all(np.isfinite(T)) or np.any(np.abs(M) > 1e154)

    def test_symmetrify_lower_and_upper(self):
        A = np.tril(_rng(6).standard_normal((5, 5)))
        B = A.copy()
        symmetrify(B)
        np.testing.assert_array_equal(B, np.tril(A) + np.tril(A, -1).T)
        C = A.T.copy()
        symmetrify(C, upper=True)
        np.testing.assert_array_equal(C, C.T)

    def test_backsub_both_sides_matches_explicit_inverse(self):
        """'left' is L^-T X L^-1, 'right' is L^-1 X L^-T."""
        A = _spd_with_condition(10, 1e4, seed=7)
        L = jitchol(A)
        X = _spd_with_condition(10, 10.0, seed=8)
        Li = np.linalg.inv(L)
        np.testing.assert_allclose(backsub_both_sides(L, X, "left"), Li.T @ X @ Li, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(backsub_both_sides(L, X, "right"), Li @ X @ Li.T, rtol=1e-8, atol=1e-10)


# =============================================================================
# 2. Elementwise and special functions
# =============================================================================


class TestSafeElementwise:
    HUGE = np.array([-1e308, -1e5, -1.0, 0.0, 1.0, 1e5, 1e308, np.inf])

    @pytest.mark.parametrize("fn", [safe_exp, safe_square, safe_cube, safe_quad, safe_three_times])
    def test_never_overflows_and_is_monotone_on_positive_side(self, fn):
        out = fn(self.HUGE[self.HUGE >= 0])
        assert np.all(np.isfinite(out))
        assert np.all(np.diff(out) >= 0)

    def test_safe_exp_exact_in_normal_range(self):
        f = np.linspace(-700, 700, 101)
        np.testing.assert_array_equal(safe_exp(f), np.exp(f))

    def test_chain_rules_finite_with_huge_inner_derivative(self):
        one = np.array([1.0])
        big = np.array([1e200])
        assert np.all(np.isfinite(chain_2(one, big, one, one)))
        assert np.all(np.isfinite(chain_3(one, big, one, one, one, one)))


class TestNormalSpecialFunctions:
    Z = [-1e3, -200.0, -38.0, -10.0, -5.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 5.0, 10.0, 38.0, 200.0]

    @pytest.mark.parametrize("z", Z)
    def test_logCdfNormal_matches_scipy(self, z):
        np.testing.assert_allclose(ug.logCdfNormal(z), sps.log_ndtr(z), rtol=1e-12, atol=1e-300)

    @pytest.mark.parametrize("z", Z)
    def test_cdfNormal_matches_scipy(self, z):
        np.testing.assert_allclose(ug.cdfNormal(z), sps.ndtr(z), rtol=1e-12, atol=1e-300)

    @pytest.mark.parametrize("z", Z)
    def test_derivLogCdfNormal_matches_inverse_mills_ratio(self, z):
        ref = np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - sps.log_ndtr(z))
        got = ug.derivLogCdfNormal(z)
        assert np.isfinite(got)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-300)

    def test_derivLogCdfNormal_tail_asymptote(self):
        # phi(z)/Phi(z) ~ -z for z -> -inf; must not blow up or collapse to 0
        for z in [-1e2, -1e4, -1e6]:
            np.testing.assert_allclose(ug.derivLogCdfNormal(z), -z, rtol=1e-3)

    def test_std_norm_pdf_handles_huge_and_infinite(self):
        x = np.array([-np.inf, -1e300, -40.0, 0.0, 40.0, 1e300, np.inf])
        p = ug.std_norm_pdf(x)
        assert not np.any(np.isnan(p))
        np.testing.assert_allclose(p, stats.norm.pdf(x), atol=1e-300)

    def test_inv_std_norm_cdf_approximation_quality(self):
        """Winitzki approximation: absolute error ~1e-2 out to p=1e-12, and odd symmetry."""
        p = np.array([1e-12, 1e-6, 1e-3, 0.1, 0.3, 0.5])
        x = ug.inv_std_norm_cdf(p)
        np.testing.assert_allclose(x, sps.ndtri(p), atol=2e-2)
        np.testing.assert_allclose(ug.inv_std_norm_cdf(1 - p[1:]), -x[1:], atol=1e-6)


# =============================================================================
# 3. Link functions
# =============================================================================

F_GRID = np.array([-1e4, -750.0, -40.0, -5.0, -1e-6, 0.0, 1e-6, 5.0, 40.0, 750.0, 1e4])[:, None]
SMOOTH_LINKS = [
    lf.Identity(),
    lf.Probit(),
    lf.ScaledProbit(nu=2.0),
    lf.Cloglog(),
    lf.Log(),
    lf.Log_ex_1(),
]


@pytest.mark.parametrize("link", SMOOTH_LINKS, ids=lambda link: type(link).__name__)
@pytest.mark.parametrize("method", ["transf", "dtransf_df", "d2transf_df2", "d3transf_df3"])
def test_link_and_derivatives_finite_over_wide_range(link, method):
    out = getattr(link, method)(F_GRID)
    assert np.all(np.isfinite(out)), F_GRID[~np.isfinite(out)].ravel()


@pytest.mark.parametrize(
    "link", [lf.Probit(), lf.ScaledProbit(nu=0.5), lf.Cloglog()], ids=lambda link: type(link).__name__
)
def test_probability_links_stay_in_unit_interval_and_monotone(link):
    f = np.linspace(-50, 50, 2001)
    p = link.transf(f)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= 0)


def test_probit_tail_matches_scipy():
    f = np.array([-37.0, -20.0, -8.0, 0.0, 8.0])
    np.testing.assert_allclose(lf.Probit().transf(f), sps.ndtr(f), rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="Log_ex_1.transf = log1p(safe_exp(f)); safe_exp clips at log(DBL_MAX) so softplus saturates "
    "at 709.78 for f > 709 instead of returning ~f (use np.logaddexp(0, f))",
)
def test_softplus_asymptote_for_large_f():
    f = np.array([30.0, 710.0, 1000.0, 1e4])
    np.testing.assert_allclose(lf.Log_ex_1().transf(f), np.logaddexp(0, f), rtol=1e-12)


def test_softplus_small_f_is_accurate():
    f = np.array([-40.0, -20.0, -1.0, 0.0])
    np.testing.assert_allclose(lf.Log_ex_1().transf(f), np.logaddexp(0, f), rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="Cloglog.transf = 1 - exp(-exp(f)) cancels catastrophically for f < ~-20 "
    "(returns 0 at f=-40 instead of 4.2e-18); use -expm1(-exp(f))",
)
def test_cloglog_relative_accuracy_in_lower_tail():
    f = np.array([-40.0, -20.0, -5.0, 0.0])
    np.testing.assert_allclose(lf.Cloglog().transf(f), -np.expm1(-np.exp(f)), rtol=1e-10)


# =============================================================================
# 4. Likelihoods
# =============================================================================

DERIVS = ["logpdf", "dlogpdf_df", "d2logpdf_df2", "d3logpdf_df3"]


def _assert_lik_finite(lik, f, y):
    for m in DERIVS:
        out = getattr(lik, m)(f, y)
        assert np.all(np.isfinite(out)), f"{m} non-finite at f={f[~np.isfinite(out)].ravel()}"


class TestBernoulli:
    def test_logpdf_matches_log_ndtr_in_core_range(self):
        f = np.linspace(-5, 5, 41)[:, None]
        lik = GPy.likelihoods.Bernoulli()
        # log p -> 0 as p -> 1, so absolute error is the meaningful measure there
        np.testing.assert_allclose(lik.logpdf(f, np.ones_like(f)), sps.log_ndtr(f), rtol=1e-10, atol=1e-15)
        # y=0 goes through 1 - Phi(f), which already sheds ~7 digits by f=5
        np.testing.assert_allclose(lik.logpdf(f, np.zeros_like(f)), sps.log_ndtr(-f), rtol=1e-8)

    @pytest.mark.xfail(
        strict=True,
        reason="Bernoulli.logpdf_link clips p at 1e-9, so log p(y|f) is floored at -20.7 for misclassified "
        "points beyond |f|~6 (exact value at f=-10 is -53.2); probit should use log_ndtr",
    )
    def test_logpdf_tail_accuracy(self):
        f = np.array([-30.0, -10.0, -8.0])[:, None]
        lik = GPy.likelihoods.Bernoulli()
        np.testing.assert_allclose(lik.logpdf(f, np.ones_like(f)), sps.log_ndtr(f), rtol=1e-8)

    @pytest.mark.xfail(
        strict=True,
        reason="Same 1e-9 clip makes dlogpdf_df vanish (7.7e-14 at f=-10, exact 10.1) for confidently "
        "misclassified points, so Laplace mode-finding gets no signal from them",
    )
    def test_gradient_tail_is_inverse_mills_ratio(self):
        f = np.array([-10.0, -8.0])[:, None]
        lik = GPy.likelihoods.Bernoulli()
        exact = np.exp(-0.5 * f**2 - 0.5 * np.log(2 * np.pi) - sps.log_ndtr(f))
        np.testing.assert_allclose(lik.dlogpdf_df(f, np.ones_like(f)), exact, rtol=1e-6)

    @pytest.mark.xfail(
        strict=True,
        reason="d3logpdf_dlink3 divides by (1-p)**3 without the clip used by the first two derivatives; "
        "-inf once Phi(f) rounds to 1 (f > ~8.3, y=0)",
    )
    def test_all_derivatives_finite_for_confident_mislabels(self):
        f = np.array([[-40.0], [40.0]])
        y = np.array([[1.0], [0.0]])
        _assert_lik_finite(GPy.likelihoods.Bernoulli(), f, y)

    def test_first_two_derivatives_finite_everywhere(self):
        lik = GPy.likelihoods.Bernoulli()
        for y in (0.0, 1.0):
            Y = np.full_like(F_GRID, y)
            for m in DERIVS[:3]:
                assert np.all(np.isfinite(getattr(lik, m)(F_GRID, Y)))

    @pytest.mark.parametrize("log_tau", np.linspace(-8, 8, 9))
    @pytest.mark.parametrize("z", [-60.0, -10.0, -1.0, 0.0, 1.0, 10.0, 60.0])
    @pytest.mark.parametrize("y", [0.0, 1.0])
    def test_ep_moment_match_valid(self, log_tau, z, y):
        """Tilted variance must be finite, positive and not exceed the cavity variance."""
        tau = 10.0**log_tau
        v = z * np.sqrt(tau**2 + tau)
        Z, mu, s2 = GPy.likelihoods.Bernoulli().moments_match_ep(np.array(y), tau, v)
        assert np.isfinite(mu) and np.isfinite(s2)
        assert 0 < s2 <= (1.0 / tau) * (1 + 1e-9)
        assert 0 <= Z <= 1

    def test_ep_moment_match_tail_variance_asymptote(self):
        # For z -> -inf the tilted variance tends to (1/tau) * tau/(tau^2+tau) / z^2 * (tau^2+tau)/tau ...
        # i.e. s2 * tau ~ 1/z^2 when tau << 1.
        tau, z = 1e-8, -1e3
        v = -z * np.sqrt(tau**2 + tau)
        _, _, s2 = GPy.likelihoods.Bernoulli().moments_match_ep(np.array(0.0), tau, v)
        np.testing.assert_allclose(s2 * tau, 1 / z**2, rtol=5e-2)

    @pytest.mark.xfail(
        strict=True,
        reason="sigma2_hat = 1/tau - r/(tau^2+tau)*(z+r) cancels catastrophically for |z| >~ 3e3; "
        "it turns negative by |z|=1e5 (vague cavity, far-away mean)",
    )
    def test_ep_moment_match_extreme_cavity_keeps_positive_variance(self):
        tau = 1e-8
        v = 1e5 * np.sqrt(tau**2 + tau)
        _, _, s2 = GPy.likelihoods.Bernoulli().moments_match_ep(np.array(0.0), tau, v)
        assert 0 < s2 <= 1 / tau

    def test_variational_expectations_finite_for_extreme_moments(self):
        lik = GPy.likelihoods.Bernoulli()
        m = np.array([-1e3, -30.0, 0.0, 30.0, 1e3])[:, None]
        v = np.array([1e-10, 1e-3, 1.0, 1e3, 1e6])[:, None]
        for y in (0.0, 1.0):
            F, dm, dv, _ = lik.variational_expectations(np.full_like(m, y), m, v)
            assert np.all(np.isfinite(F)) and np.all(np.isfinite(dm)) and np.all(np.isfinite(dv))
            assert np.all(F <= 0)


class TestCountAndPositiveLikelihoods:
    def test_poisson_logpdf_matches_scipy(self):
        f = np.linspace(-30, 30, 13)[:, None]
        y = np.array([0, 1, 3, 10, 100, 1000, 0, 1, 5, 20, 50, 7, 1e4])[:, None]
        lik = GPy.likelihoods.Poisson()
        np.testing.assert_allclose(lik.logpdf(f, y), stats.poisson.logpmf(y, np.exp(f)), rtol=1e-10)

    def test_poisson_large_counts_finite(self):
        f = np.log(np.array([[1e6], [1e9]]))
        _assert_lik_finite(GPy.likelihoods.Poisson(), f, np.round(np.exp(f)))

    @pytest.mark.xfail(
        strict=True,
        reason="Poisson.logpdf_link computes y*log(exp(f)); exp underflows to 0 below f~-745 giving "
        "0*-inf=NaN (y=0) or -inf (y>0). Working in f directly (y*f - exp(f)) avoids it",
    )
    def test_poisson_deep_negative_latent(self):
        f = np.array([[-800.0], [-800.0]])
        y = np.array([[0.0], [1.0]])
        _assert_lik_finite(GPy.likelihoods.Poisson(), f, y)

    def test_gamma_logpdf_matches_scipy_moderate(self):
        f = np.array([[-2.0], [0.0], [2.0], [4.5]])  # alpha = beta*exp(f) stays below 171
        y = np.array([[0.3], [1.0], [5.0], [100.0]])
        beta = 1.5
        lik = GPy.likelihoods.Gamma(beta=beta)
        alpha = np.exp(f) * beta
        np.testing.assert_allclose(lik.logpdf(f, y), stats.gamma.logpdf(y, alpha, scale=1 / beta), rtol=1e-10)

    @pytest.mark.xfail(
        strict=True,
        reason="Gamma.logpdf_link uses log(special.gamma(alpha)) which overflows for alpha > 171.6 "
        "(f > ~5.1 with beta=1, i.e. ordinary data y >~ 170); use special.gammaln",
    )
    def test_gamma_logpdf_finite_for_large_shape(self):
        f = np.array([[5.2], [6.0], [10.0]])
        y = np.exp(f)
        lik = GPy.likelihoods.Gamma(beta=1.0)
        np.testing.assert_allclose(lik.logpdf(f, y), stats.gamma.logpdf(y, np.exp(f)), rtol=1e-8)

    def test_student_t_logpdf_matches_scipy_for_huge_residuals(self):
        lik = GPy.likelihoods.StudentT(deg_free=3.0, sigma2=0.5)
        y = np.array([-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6])[:, None]
        f = np.zeros_like(y)
        np.testing.assert_allclose(lik.logpdf(f, y), stats.t.logpdf(y, 3.0, scale=np.sqrt(0.5)), rtol=1e-10)
        _assert_lik_finite(lik, f, y)

    @pytest.mark.parametrize("variance", [1e-300, 1e-12, 1.0, 1e12, 1e300])
    def test_gaussian_logpdf_finite_at_extreme_variance(self, variance):
        lik = GPy.likelihoods.Gaussian(variance=variance)
        y = np.array([[-1e3], [0.0], [1e3]])
        _assert_lik_finite(lik, np.zeros_like(y), y)

    @pytest.mark.parametrize(
        "lik",
        [GPy.likelihoods.Exponential(), GPy.likelihoods.Weibull(), GPy.likelihoods.LogLogistic()],
        ids=lambda link: type(link).__name__,
    )
    def test_positive_likelihoods_finite_in_working_range(self, lik):
        f = np.linspace(-20, 20, 9)[:, None]
        y = np.array([1e-8, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3, 1e6])[:, None]
        _assert_lik_finite(lik, f, y)


# =============================================================================
# 5. Kernels
# =============================================================================

STATIONARY = [
    GPy.kern.RBF,
    GPy.kern.Exponential,
    GPy.kern.OU,
    GPy.kern.Matern32,
    GPy.kern.Matern52,
    GPy.kern.ExpQuad,
    GPy.kern.RatQuad,
]
LENGTHSCALES = [1e-6, 1e-2, 1.0, 1e3, 1e8]


def _inputs_with_duplicates(d=2, n=30, seed=0):
    X = _rng(seed).uniform(0, 1, (n, d))
    return np.vstack([X, X[:5]])  # exact duplicates: r == 0 off the diagonal


def _check_kernel_matrix(k, X):
    K = k.K(X)
    assert np.all(np.isfinite(K))
    np.testing.assert_allclose(K, K.T, rtol=1e-14, atol=1e-300)
    np.testing.assert_allclose(k.Kdiag(X), np.diag(K), rtol=1e-12, atol=1e-300)
    scale = max(np.abs(K).max(), 1e-300)
    assert np.linalg.eigvalsh(K).min() >= -1e-10 * scale


def _check_kernel_gradients(k, X, with_X_grads=True):
    n = X.shape[0]
    k.update_gradients_full(np.ones((n, n)), X)
    assert np.all(np.isfinite(k.gradient))
    if with_X_grads:
        k.update_gradients_diag(np.ones(n), X)
        assert np.all(np.isfinite(k.gradient))
        assert np.all(np.isfinite(k.gradients_X(np.ones((n, n)), X)))
        assert np.all(np.isfinite(k.gradients_X_diag(np.ones(n), X)))


@pytest.mark.parametrize("kern_cls", STATIONARY, ids=lambda c: c.__name__)
@pytest.mark.parametrize("ls", LENGTHSCALES)
@pytest.mark.parametrize("ard", [False, True])
def test_stationary_kernel_psd_symmetric_and_finite_gradients(kern_cls, ls, ard):
    X = _inputs_with_duplicates()
    k = kern_cls(2, ARD=ard)
    k.lengthscale[:] = ls
    _check_kernel_matrix(k, X)
    _check_kernel_gradients(k, X)


@pytest.mark.parametrize(
    "make",
    [
        lambda: GPy.kern.Linear(2),
        lambda: GPy.kern.Linear(2, ARD=True),
        lambda: GPy.kern.MLP(2),
        lambda: GPy.kern.White(2),
        lambda: GPy.kern.Bias(2),
        lambda: GPy.kern.StdPeriodic(2),
        lambda: GPy.kern.RBF(2) + GPy.kern.White(2),
        lambda: GPy.kern.RBF(2) * GPy.kern.Linear(2),
    ],
    ids=["Linear", "LinearARD", "MLP", "White", "Bias", "StdPeriodic", "RBF+White", "RBF*Linear"],
)
def test_other_kernels_psd_symmetric_and_finite_gradients(make):
    X = _inputs_with_duplicates()
    k = make()
    _check_kernel_matrix(k, X)
    _check_kernel_gradients(k, X)


def test_poly_kernel_psd_and_param_gradients():
    X = _inputs_with_duplicates()
    k = GPy.kern.Poly(2)
    _check_kernel_matrix(k, X)
    _check_kernel_gradients(k, X, with_X_grads=False)  # diag and X gradients not implemented


@pytest.mark.parametrize("ls", [1e-2, 1.0, 1e3])
def test_cosine_kernel_psd_in_one_dimension(ls):
    """cos(r) is only positive-definite on R^1; that is the regime we test."""
    X = _rng(1).uniform(0, 5, (30, 1))
    k = GPy.kern.Cosine(1, lengthscale=ls)
    _check_kernel_matrix(k, X)


SMOOTH_AT_ZERO = [GPy.kern.RBF, GPy.kern.Matern32, GPy.kern.Matern52, GPy.kern.ExpQuad, GPy.kern.RatQuad]
ROUGH_AT_ZERO = [GPy.kern.Exponential, GPy.kern.OU]


@pytest.mark.parametrize(
    "kern_cls",
    SMOOTH_AT_ZERO
    + [
        pytest.param(
            c,
            marks=pytest.mark.xfail(
                strict=True,
                reason="K(X, X2) has no r=0 fix-up: the expanded r^2 leaves ~1e-16 at coincident points, "
                "sqrt turns it into r~1e-8, and kernels with a cusp at 0 pass that straight through "
                "(k = 1 - 2e-8). Affects predictions at training inputs",
            ),
        )
        for c in ROUGH_AT_ZERO
    ],
    ids=lambda c: c.__name__,
)
def test_K_with_X2_equal_X_is_consistent(kern_cls):
    X = _inputs_with_duplicates()
    k = kern_cls(2)
    np.testing.assert_allclose(k.K(X, X), k.K(X), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diag(k.K(X, X)), k.variance.values[0], rtol=1e-12)


@pytest.mark.parametrize("kern_cls", [GPy.kern.RBF, GPy.kern.Matern32, GPy.kern.Exponential], ids=lambda c: c.__name__)
def test_stationary_translation_invariance_modest_offset(kern_cls):
    x = np.linspace(0, 1, 20)[:, None]
    k = kern_cls(1)
    np.testing.assert_allclose(k.K(x + 1e3), k.K(x), atol=1e-8)


@pytest.mark.xfail(
    strict=True,
    reason="Stationary._unscaled_dist uses |x|^2 + |x'|^2 - 2x.x', which cancels catastrophically when "
    "inputs share a large offset (error ~1e-3 at 1e6, ~0.8 at 1e8; Unix timestamps are ~1.7e9). "
    "Centering X or computing differences directly fixes it",
)
@pytest.mark.parametrize("kern_cls", [GPy.kern.RBF, GPy.kern.Matern32, GPy.kern.Exponential], ids=lambda c: c.__name__)
@pytest.mark.parametrize("offset", [1e6, 1.7e9])
def test_stationary_translation_invariance_large_offset(kern_cls, offset):
    x = np.linspace(0, 1, 20)[:, None]
    k = kern_cls(1)
    np.testing.assert_allclose(k.K(x + offset), k.K(x), atol=1e-6)
    # the X2 path has no diagonal fix-up, so coincident points drift from k(0) too
    np.testing.assert_allclose(np.diag(k.K(x + offset, x + offset)), k.variance.values[0], atol=1e-6)


# =============================================================================
# 6. Models and inference
# =============================================================================


class TestExactGPRegression:
    def test_duplicate_inputs_with_tiny_noise(self):
        X, Y = _regression_data()
        m = GPy.models.GPRegression(np.vstack([X] * 3), np.vstack([Y] * 3))
        m.Gaussian_noise.variance = 1e-10
        assert _all_finite(m)

    @pytest.mark.parametrize("noise", [1e-300, 1e-12])
    def test_vanishing_noise(self, noise):
        X, Y = _regression_data()
        m = GPy.models.GPRegression(X, Y)
        m.Gaussian_noise.variance = noise
        assert _all_finite(m)

    def test_predictive_variance_nonnegative_and_covariance_psd(self):
        X, Y = _regression_data()
        m = GPy.models.GPRegression(X, Y)
        m.Gaussian_noise.variance = 1e-8
        m.kern.lengthscale = 5.0  # strongly correlated training points
        Xs = np.vstack([X, np.linspace(-5, 15, 50)[:, None]])
        _, v = m.predict_noiseless(Xs)
        assert np.all(v >= 0)
        _, C = m.predict_noiseless(Xs, full_cov=True)
        np.testing.assert_allclose(C, C.T, atol=1e-12)
        assert np.linalg.eigvalsh(C).min() >= -1e-8 * np.abs(C).max()

    def test_constant_targets_with_normalizer(self):
        X, Y = _regression_data()
        with pytest.warns(UserWarning, match="standard deviation of zero"):
            m = GPy.models.GPRegression(X, np.full_like(Y, 3.0), normalizer=True)
        assert _all_finite(m)
        m.optimize(max_iters=50)
        assert _all_finite(m)
        mu, var = m.predict(X[:5])
        np.testing.assert_allclose(mu, 3.0, atol=1e-6)
        assert np.all(np.isfinite(var)) and np.all(var >= 0)

    def test_normalizer_equivariance_to_affine_rescaling_of_Y(self):
        X, Y = _regression_data()
        a, b = 1e8, 1e9
        m1 = GPy.models.GPRegression(X, Y, normalizer=True)
        m2 = GPy.models.GPRegression(X, a * Y + b, normalizer=True)
        m1.optimize()
        m2.optimize()
        np.testing.assert_allclose(m2.param_array, m1.param_array, rtol=1e-5)
        mu1, v1 = m1.predict(X)
        mu2, v2 = m2.predict(X)
        np.testing.assert_allclose((mu2 - b) / a, mu1, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(v2 / a**2, v1, rtol=1e-6)

    @pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
    def test_input_rescaling_invariance(self, c):
        X, Y = _regression_data()
        ma = GPy.models.GPRegression(X, Y)
        ma.kern.lengthscale = 2.0
        mb = GPy.models.GPRegression(X * c, Y)
        mb.kern.lengthscale = 2.0 * c
        np.testing.assert_allclose(mb.log_likelihood(), ma.log_likelihood(), rtol=1e-10)

    @pytest.mark.parametrize(
        "ls,var,noise",
        [
            (1e-3, 1.0, 1e-2),
            (1e3, 1.0, 1e-2),
            (1.0, 1e-6, 1e-2),
            (1.0, 1e6, 1e-2),
            (1.0, 1.0, 1e-6),
            (1.0, 1.0, 1e3),
        ],
    )
    def test_gradients_correct_at_extreme_hyperparameters(self, ls, var, noise):
        X, Y = _regression_data()
        m = GPy.models.GPRegression(X, Y)
        m.kern.lengthscale = ls
        m.kern.variance = var
        m.Gaussian_noise.variance = noise
        assert _all_finite(m)
        assert m.checkgrad()

    @pytest.mark.parametrize("kern_cls", [GPy.kern.Exponential, GPy.kern.Matern32, GPy.kern.Matern52])
    def test_rough_kernels_gradients_with_duplicate_inputs(self, kern_cls):
        """Non-differentiable-at-zero kernels must still give finite, correct hyper gradients."""
        X, Y = _regression_data(n=20)
        m = GPy.models.GPRegression(np.vstack([X, X[:5]]), np.vstack([Y, Y[:5]]), kernel=kern_cls(1))
        assert _all_finite(m)
        assert m.checkgrad()

    def test_noise_at_transform_floor_gives_finite_gradient(self):
        X, Y = _regression_data()
        m = GPy.models.GPRegression(X, Y)
        oa = m.optimizer_array.copy()
        oa[-1] = -800.0  # Logexp floor for the noise variance
        m.optimizer_array = oa
        assert _all_finite(m)

    @pytest.mark.xfail(
        strict=True,
        reason="With every positive parameter at the Logexp floor (~5.6e-309) the objective is finite but "
        "the lengthscale gradient is NaN (0*inf), which poisons quasi-Newton optimisers",
    )
    def test_all_parameters_at_transform_floor(self):
        X, Y = _regression_data()
        m = GPy.models.GPRegression(X, Y)
        m.optimizer_array = np.full_like(m.optimizer_array, -800.0)
        assert np.isfinite(m.objective_function())
        assert np.all(np.isfinite(m.objective_function_gradients()))


class TestSparseAndStochastic:
    def test_coincident_inducing_inputs(self):
        X, Y = _regression_data()
        m = GPy.models.SparseGPRegression(X, Y, Z=np.repeat(X[:1], 10, axis=0))
        assert _all_finite(m)

    def test_inducing_equal_to_inputs_recovers_exact_gp(self):
        X, Y = _regression_data()
        s = GPy.models.SparseGPRegression(X, Y, Z=X.copy())
        f = GPy.models.GPRegression(X, Y)
        s.Gaussian_noise.variance = f.Gaussian_noise.variance = 1e-2
        np.testing.assert_allclose(np.asarray(s.log_likelihood()).item(), f.log_likelihood(), rtol=1e-5)

    @pytest.mark.parametrize("ls", [1e-3, 1e4])
    def test_extreme_lengthscale(self, ls):
        X, Y = _regression_data()
        m = GPy.models.SparseGPRegression(X, Y, num_inducing=10)
        m.kern.lengthscale = ls
        assert _all_finite(m)

    def test_svgp_extreme_lengthscale(self):
        pytest.importorskip("climin")  # SVGP's optimiser dependency is optional
        rng = _rng(3)
        X = rng.uniform(0, 10, (200, 1))
        m = GPy.core.SVGP(X, np.sin(X), X[:10].copy(), GPy.kern.RBF(1), GPy.likelihoods.Gaussian(), batchsize=50)
        m.kern.lengthscale = 1e4
        assert _all_finite(m)


def _separable_classification(seed=2):
    rng = _rng(seed)
    X = np.vstack([rng.uniform(-3, -1, (20, 1)), rng.uniform(1, 3, (20, 1))])
    Y = np.vstack([np.zeros((20, 1)), np.ones((20, 1))])
    return X, Y


def _classifier(kind, X, Y):
    if kind == "EP":
        return GPy.models.GPClassification(X, Y)
    return GPy.core.GP(
        X, Y, kernel=GPy.kern.RBF(1), likelihood=GPy.likelihoods.Bernoulli(), inference_method=LaplaceInf()
    )


class TestNonGaussianInference:
    @pytest.mark.parametrize("kind", ["EP", "Laplace"])
    def test_separable_data_with_large_kernel_variance(self, kind):
        X, Y = _separable_classification()
        m = _classifier(kind, X, Y)
        m.kern.variance = 1e4
        m.kern.lengthscale = 10.0
        assert _all_finite(m)

    @pytest.mark.parametrize("kind", ["EP", "Laplace"])
    def test_optimised_classifier_predictions_are_probabilities(self, kind):
        X, Y = _separable_classification()
        m = _classifier(kind, X, Y)
        m.optimize(max_iters=200)
        assert _all_finite(m)
        p, _ = m.predict(np.linspace(-10, 10, 50)[:, None])
        assert np.all((p >= 0) & (p <= 1))

    @pytest.mark.parametrize("kind", ["EP", "Laplace"])
    def test_mislabelled_point_with_large_kernel_variance(self, kind):
        X, Y = _separable_classification()
        Y[0] = 1
        m = _classifier(kind, X, Y)
        m.kern.variance = 1e3
        assert _all_finite(m)

    def test_poisson_laplace_all_zero_counts(self):
        X, _ = _regression_data()
        m = GPy.core.GP(X, np.zeros_like(X), GPy.kern.RBF(1), GPy.likelihoods.Poisson(), inference_method=LaplaceInf())
        assert _all_finite(m)
        m.optimize(max_iters=100)
        assert _all_finite(m)

    def test_poisson_laplace_large_counts(self):
        X, _ = _regression_data()
        Y = _rng(4).poisson(1e4, X.shape).astype(float)
        m = GPy.core.GP(X, Y, GPy.kern.RBF(1), GPy.likelihoods.Poisson(), inference_method=LaplaceInf())
        assert _all_finite(m)

    def test_student_t_laplace_with_gross_outliers(self):
        X, Y = _regression_data()
        Y = Y.copy()
        Y[::7] += 1e4
        lik = GPy.likelihoods.StudentT(deg_free=3.0, sigma2=0.01)
        m = GPy.core.GP(X, Y, GPy.kern.RBF(1), lik, inference_method=LaplaceInf())
        assert _all_finite(m)

    @pytest.mark.parametrize("sigma2", [0.1, 1.0])
    def test_student_t_laplace_gradients(self, sigma2):
        X, Y = _regression_data()
        lik = GPy.likelihoods.StudentT(deg_free=5.0, sigma2=sigma2)
        m = GPy.core.GP(X, Y, GPy.kern.RBF(1), lik, inference_method=LaplaceInf())
        assert m.checkgrad()

    @pytest.mark.xfail(
        strict=True,
        reason="Gamma likelihood overflows log(Gamma(alpha)) for y >~ 170, the Laplace objective turns NaN, "
        "and laplace.rasm_mode then hits a leftover `import ipdb; ipdb.set_trace()`",
    )
    def test_gamma_laplace_on_ordinary_positive_data(self, monkeypatch):
        # Guard: if ipdb is installed, the leftover breakpoint would hang a non-interactive run.
        fake = types.ModuleType("ipdb")

        def _no_debugger(*_a, **_k):
            raise RuntimeError("debugger breakpoint reached")

        fake.set_trace = _no_debugger
        monkeypatch.setitem(sys.modules, "ipdb", fake)

        X, _ = _regression_data()
        Y = np.exp(_rng(5).uniform(5, 7, X.shape))  # ~150 .. 1100
        m = GPy.core.GP(X, Y, GPy.kern.RBF(1), GPy.likelihoods.Gamma(), inference_method=LaplaceInf())
        assert _all_finite(m)


# =============================================================================
# 7. Source hygiene
# =============================================================================


def _breakpoint_lines(source):
    """Line numbers of executable set_trace()/breakpoint() calls (comments are ignored by ast)."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in {"set_trace", "breakpoint"}:
                lines.append(node.lineno)
    return sorted(lines)


def _live_breakpoints():
    root = pathlib.Path(GPy.__file__).parent
    hits = []
    for path in root.rglob("*.py"):
        if "testing" in path.parts:
            continue
        try:
            lines = _breakpoint_lines(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        hits += [f"{path.relative_to(root.parent)}:{n}" for n in lines]
    return sorted(hits)


def test_breakpoint_scanner_self_check():
    src = "import pdb\n# pdb.set_trace()\nif x:\n    pdb.set_trace()\nbreakpoint()\n"
    assert _breakpoint_lines(src) == [4, 5]


@pytest.mark.xfail(
    strict=True,
    reason="Debugger breakpoints remain on numerical error paths: laplace.py (NaN objective), "
    "var_dtc.py (VVT_factor shape branch), state_space_main.py",
)
def test_no_live_debugger_breakpoints_in_package():
    assert _live_breakpoints() == []
