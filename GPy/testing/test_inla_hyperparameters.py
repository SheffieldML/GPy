"""CIP-0015: INLA-style hyperparameter integration."""

import numpy as np
import pytest

import GPy
from GPy.inference.hyperparameters import INLAHyperparameters
from GPy.inference.hyperparameters.inla import _normalised_weights


def _classifier(n=25, seed=1, priors=True):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, (n, 1)), axis=0)
    Y = (rng.uniform(size=(n, 1)) < 1 / (1 + np.exp(-3 * np.sin(1.5 * X)))).astype(float)
    k = GPy.kern.RBF(1)
    if priors:
        k.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
        k.variance.set_prior(GPy.priors.LogGaussian(1.0, 1.0), warning=False)
    return GPy.core.GP(
        X,
        Y,
        kernel=k,
        likelihood=GPy.likelihoods.Bernoulli(),
        inference_method=GPy.inference.latent_function_inference.Laplace(),
    )


def _regressor(lengthscale_prior_sd=1.0):
    """1-D hyperparameter problem (lengthscale only) with a cheap exact likelihood."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-3, 3, (20, 1))
    Y = np.sin(X) + 0.2 * rng.standard_normal((20, 1))
    k = GPy.kern.RBF(1)
    k.variance.fix(1.0)
    k.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, lengthscale_prior_sd), warning=False)
    lik = GPy.likelihoods.Gaussian(variance=0.04)
    lik.variance.fix()
    return GPy.core.GP(X, Y, kernel=k, likelihood=lik)


def test_weights_normalised_and_model_restored():
    m = _classifier()
    inla = INLAHyperparameters(m).fit()
    assert np.isclose(inla.weights.sum(), 1.0)
    assert np.all(inla.weights >= 0)
    assert inla.design_used == "grid"
    np.testing.assert_allclose(m.optimizer_array, inla.u_star)
    Xs = np.linspace(-3, 3, 5)[:, None]
    inla.predict_noiseless(Xs)
    np.testing.assert_allclose(m.optimizer_array, inla.u_star)


def test_mixture_variance_covers_plugin_spread():
    m = _classifier()
    inla = INLAHyperparameters(m).fit()
    Xs = np.linspace(-3.5, 3.5, 8)[:, None]
    mu_map, var_map = m.predict_noiseless(Xs)
    mu, var = inla.predict_noiseless(Xs)
    assert mu.shape == mu_map.shape and var.shape == var_map.shape
    assert np.all(var > 0)
    assert var.mean() > var_map.mean()


@pytest.mark.parametrize("design", ["grid", "ccd", "importance"])
def test_matches_dense_quadrature_in_one_dimension(design):
    m = _regressor()
    inla = INLAHyperparameters(m, design=design, grid_step=0.5, grid_cutoff=6.0, n_samples=300).fit()
    Xs = np.linspace(-3, 3, 6)[:, None]
    mu, var = inla.predict_noiseless(Xs)

    z = np.linspace(-6, 6, 241)
    U = inla.u_star + z[:, None] @ inla.L.T
    w = _normalised_weights([inla._log_post(u) for u in U])
    mus, vs = [], []
    for u in U:
        m.optimizer_array = u
        a, b = m.predict_noiseless(Xs)
        mus.append(a)
        vs.append(b)
    mus, vs = np.array(mus), np.array(vs)
    W = w[:, None, None]
    mu_ref = (W * mus).sum(0)
    var_ref = (W * (vs + mus**2)).sum(0) - mu_ref**2
    tol = {"grid": 5e-2, "ccd": 1.5e-1, "importance": 1e-1}[design]
    np.testing.assert_allclose(mu, mu_ref, atol=tol)
    np.testing.assert_allclose(var, var_ref, atol=tol, rtol=tol)


def test_tight_prior_collapses_to_map():
    m = _regressor(lengthscale_prior_sd=1e-3)
    inla = INLAHyperparameters(m, design="grid").fit()
    Xs = np.linspace(-3, 3, 5)[:, None]
    mu_map, var_map = m.predict_noiseless(Xs)
    mu, var = inla.predict_noiseless(Xs)
    np.testing.assert_allclose(mu, mu_map, atol=1e-4)
    np.testing.assert_allclose(var, var_map, atol=1e-4)


def test_missing_prior_raises():
    m = _classifier(priors=False)
    with pytest.raises(ValueError, match="prior"):
        INLAHyperparameters(m).fit()


def test_fixed_parameters_do_not_need_priors():
    m = _regressor()
    m.kern.variance.fix(1.0)
    INLAHyperparameters(m).fit()  # only the lengthscale is free, and it has a prior


def test_indefinite_hessian_raises(monkeypatch):
    m = _classifier()
    inla = INLAHyperparameters(m)
    monkeypatch.setattr(inla, "_hessian", lambda u: -np.eye(u.size))
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        inla.fit()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        INLAHyperparameters(_classifier()).predict_noiseless(np.zeros((1, 1)))


def test_hyperparameter_summary_names_and_positivity():
    inla = INLAHyperparameters(_classifier()).fit()
    summary = inla.hyperparameter_summary()
    assert len(summary) == 2
    assert all(mean > 0 and sd > 0 for mean, sd in summary.values())


@pytest.mark.filterwarnings("ignore:Importance design has an effective sample size")
def test_importance_used_automatically_above_three_hyperparameters():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 4))
    Y = np.sin(X[:, :1]) + 0.1 * rng.standard_normal((30, 1))
    k = GPy.kern.RBF(4, ARD=True)
    k.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    k.variance.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    lik = GPy.likelihoods.Gaussian(variance=0.01)
    lik.variance.fix()
    m = GPy.core.GP(X, Y, kernel=k, likelihood=lik)
    inla = INLAHyperparameters(m).fit()
    assert inla.design_used == "importance"
    assert np.isclose(inla.weights.sum(), 1.0)
    assert 1 < inla.effective_sample_size <= len(inla.weights)


def test_invalid_design_rejected():
    with pytest.raises(ValueError):
        INLAHyperparameters(_classifier(), design="nope")


def test_hyperparameter_summary_with_fixed_parameters():
    m = _regressor()  # variance and noise are fixed, only the lengthscale is free
    summary = INLAHyperparameters(m).fit().hyperparameter_summary()
    assert list(summary) == ["gp.rbf.lengthscale"]


def test_importance_design_is_reproducible_and_antithetic():
    m = _regressor()
    a = INLAHyperparameters(m, design="importance", n_samples=60, seed=7).fit()
    b = INLAHyperparameters(m, design="importance", n_samples=60, seed=7).fit()
    c = INLAHyperparameters(m, design="importance", n_samples=60, seed=8).fit()
    np.testing.assert_allclose(a.weights, b.weights)
    assert not np.allclose(a.Z, c.Z)
    assert len(a.weights) <= 60
    # skew-aware draws mirror in sign; without skew they are exact negatives
    assert len(a.Z) == 60
    np.testing.assert_array_equal(np.sign(a.Z[:30]), -np.sign(a.Z[30:]))
    sym = INLAHyperparameters(m, design="importance", n_samples=60, seed=7, skew=False).fit()
    assert len(sym.Z) == 60
    np.testing.assert_allclose(sym.Z[:30], -sym.Z[30:])


def test_importance_effective_sample_size_warning(monkeypatch):
    m = _regressor()
    inla = INLAHyperparameters(m, design="importance", n_samples=10, proposal_scale=50.0)
    with pytest.warns(UserWarning, match="effective sample size"):
        inla.fit()


def _concentrated_ard_model():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 4))
    Y = np.sin(X[:, :1]) + 0.1 * rng.standard_normal((30, 1))
    k = GPy.kern.RBF(4, ARD=True)
    k.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    k.variance.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    lik = GPy.likelihoods.Gaussian(variance=0.01)
    lik.variance.fix()
    return GPy.core.GP(X, Y, kernel=k, likelihood=lik)


@pytest.mark.filterwarnings("ignore:Importance design has an effective sample size")
def test_skew_proposal_raises_effective_sample_size_on_skewed_posterior():
    m = _concentrated_ard_model()
    sym = INLAHyperparameters(m, design="importance", n_samples=500, skew=False).fit()
    skew = INLAHyperparameters(m, design="importance", n_samples=500, skew=True).fit()
    assert skew.effective_sample_size > 2 * sym.effective_sample_size
    plus, minus = skew.skew_scales
    assert np.max(np.abs(np.log(plus / minus))) > 0.5  # at least one axis is clearly asymmetric


def test_symmetric_importance_matches_dense_quadrature_in_one_dimension():
    m = _regressor()
    inla = INLAHyperparameters(m, design="importance", n_samples=300, skew=False).fit()
    assert not hasattr(inla, "skew_scales")
    Xs = np.linspace(-3, 3, 6)[:, None]
    mu, _ = inla.predict_noiseless(Xs)
    z = np.linspace(-6, 6, 241)
    U = inla.u_star + z[:, None] @ inla.L.T
    w = _normalised_weights([inla._log_post(u) for u in U])
    mus = []
    for u in U:
        m.optimizer_array = u
        mus.append(m.predict_noiseless(Xs)[0])
    mu_ref = (w[:, None, None] * np.array(mus)).sum(0)
    np.testing.assert_allclose(mu, mu_ref, atol=1e-1)
