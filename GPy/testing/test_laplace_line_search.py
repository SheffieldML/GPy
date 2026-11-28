"""Backlog 2026-11-28_laplace-brent-bracket-failure: Laplace line search robustness."""

import warnings

import numpy as np
import pytest

import GPy


def _binomial_periodic_model(variance, lengthscale):
    rng = np.random.default_rng(0)
    days = np.arange(1, 367)
    trials = np.full(366, 2.0)
    trials[59] = 1.0
    p = 1 / (1 + np.exp(-(-1.0 + 1.2 * np.sin(2 * np.pi * days / 366 - 1.0) + 0.6 * np.sin(4 * np.pi * days / 366))))
    Y = rng.binomial(trials.astype(int), p)[:, None].astype(float)
    kern = GPy.kern.StdPeriodic(1, period=366)
    kern.period.fix()
    m = GPy.core.GP(
        days[:, None].astype(float),
        Y,
        kernel=kern,
        likelihood=GPy.likelihoods.Binomial(),
        inference_method=GPy.inference.latent_function_inference.Laplace(),
        Y_metadata={"trials": trials[:, None]},
    )
    m.kern.variance = variance
    m.kern.lengthscale = lengthscale
    return m


@pytest.mark.parametrize("variance", [1e-8, 1e-6])
@pytest.mark.parametrize("lengthscale", [1.0, 1e3])
def test_binomial_laplace_tiny_variance_does_not_raise(variance, lengthscale):
    """A near-degenerate prior must give a finite objective, not abort via BracketError."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _binomial_periodic_model(variance, lengthscale)
        assert np.isfinite(m.log_likelihood())
