"""CIP-000F Stage 2N: condition-once / predict-many via pure NumPy state.

Demonstrates a dense ``ConditionedState`` without introducing a production
operator hierarchy. Parity target: ``GPRegression.predict``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def condition_rbf(X, Y, variance, lengthscale, noise, jitter=1e-8):
    """Return LML and a ConditionedState dict (chol, alpha, train X, params)."""
    from GPy.inference.latent_function_inference import exact_gp_ops

    lml, extras = exact_gp_ops.rbf_log_marginal(X, Y, variance, lengthscale, noise)
    return lml, {
        "X": np.asarray(X, dtype=np.float64),
        "woodbury_chol": extras["woodbury_chol"],
        "alpha": extras["alpha"],
        "variance": float(np.asarray(variance).reshape(-1)[0]),
        "lengthscale": float(np.asarray(lengthscale).reshape(-1)[0]),
        "noise": float(np.asarray(noise).reshape(-1)[0]) + jitter,
    }


def predict_from_state(state, Xnew):
    """Predictive mean and variance from conditioned Cholesky (no refactor)."""
    from GPy.kern.src import rbf_ops

    X = state["X"]
    Xnew = np.asarray(Xnew, dtype=np.float64)
    L = state["woodbury_chol"]
    alpha = state["alpha"]
    variance = state["variance"]
    lengthscale = state["lengthscale"]
    Kx = rbf_ops.K(Xnew, variance, lengthscale, X2=X, ARD=False)
    mean = Kx @ alpha
    # Solve L v = Kx.T
    from GPy.util.linalg import dtrtrs

    v, _ = dtrtrs(L, Kx.T, lower=1)
    var = np.maximum(variance - np.sum(v * v, axis=0), 0.0)[:, None]
    return mean, var


def run_parity(seed=1):
    import GPy

    rng = np.random.RandomState(seed)
    X = rng.randn(25, 1)
    Y = np.sin(X) + 0.05 * rng.randn(25, 1)
    Xnew = rng.randn(8, 1)
    kern = GPy.kern.RBF(1, variance=0.9, lengthscale=1.2)
    m = GPy.models.GPRegression(X, Y, kernel=kern)
    m.Gaussian_noise.variance = 0.04
    variance = float(np.asarray(m.kern.variance).reshape(-1)[0])
    lengthscale = float(np.asarray(m.kern.lengthscale).reshape(-1)[0])
    noise = float(np.asarray(m.Gaussian_noise.variance).reshape(-1)[0])
    Y_train = np.asarray(m.Y_normalized, dtype=float)

    _lml, state = condition_rbf(X, Y_train, variance, lengthscale, noise)
    mean_s, var_s = predict_from_state(state, Xnew)
    mean_m, var_m = m.predict(Xnew, include_likelihood=False)
    return {
        "mean_abs_err": float(np.max(np.abs(mean_s - mean_m))),
        "var_abs_err": float(np.max(np.abs(var_s - var_m))),
    }


def main():
    r = run_parity()
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
