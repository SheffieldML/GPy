"""ExactGaussianInference adds ``jitter`` to the noise variance; jitter=0 evaluates the stated model."""

import numpy as np
import GPy
from GPy.inference.latent_function_inference.exact_gaussian_inference import ExactGaussianInference


def _model(jitter=None, noise=1e-3):
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    Y = np.sin(3 * X[:, :1]) + 0.05 * rng.randn(30, 1)
    inf = ExactGaussianInference() if jitter is None else ExactGaussianInference(jitter=jitter)
    return GPy.core.GP(X, Y, GPy.kern.RBF(2), GPy.likelihoods.Gaussian(variance=noise), inference_method=inf), X, Y


def _reference(X, Y, noise, variance=1.0, lengthscale=1.0):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Ky = variance * np.exp(-0.5 * d2 / lengthscale**2) + noise * np.eye(len(X))
    _, logdet = np.linalg.slogdet(Ky)
    return -0.5 * (len(X) * np.log(2 * np.pi) + logdet + Y.T @ np.linalg.solve(Ky, Y)).item()


def test_default_is_the_historical_value():
    assert ExactGaussianInference().jitter == 1e-8
    m, X, Y = _model()
    np.testing.assert_allclose(m.log_likelihood(), _reference(X, Y, 1e-3 + 1e-8), rtol=1e-12)


def test_zero_jitter_evaluates_the_stated_model():
    m, X, Y = _model(jitter=0.0)
    np.testing.assert_allclose(m.log_likelihood(), _reference(X, Y, 1e-3), rtol=1e-12)
    assert abs(m.log_likelihood() - _model()[0].log_likelihood()) > 1e-9  # the default really differs


def test_gradient_follows_the_chosen_jitter():
    m, _, _ = _model(jitter=0.0)
    assert m.checkgrad()


def test_serialization_records_the_jitter_explicitly():
    assert ExactGaussianInference().to_dict()["jitter"] == 1e-8
    d = ExactGaussianInference(jitter=0.0).to_dict()
    assert d["jitter"] == 0.0
    assert ExactGaussianInference.from_dict(d).jitter == 0.0
    assert ExactGaussianInference.from_dict(ExactGaussianInference().to_dict()).jitter == 1e-8
