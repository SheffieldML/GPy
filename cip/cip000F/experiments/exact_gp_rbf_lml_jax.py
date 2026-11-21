"""CIP-000F Stage 2T: JAX exact-GP RBF log marginal (out of package).

Not part of the GPy runtime. Requires an optional ``jax`` install; does not
modify GPy package metadata. Prefer::

    uv run --with 'jax[cpu]' python cip/cip000F/experiments/exact_gp_rbf_lml_jax.py

or run via the gated pytest in ``test_exact_gp_rbf_lml_jax.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow importing GPy from the repo root when run as a script.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

LOG_2_PI = math.log(2.0 * math.pi)


def numpy_rbf_lml(X, Y, variance, lengthscale, noise, jitter=1e-8):
    """GPy Stage 2 reference (pure NumPy)."""
    from GPy.inference.latent_function_inference import exact_gp_ops

    return exact_gp_ops.rbf_log_marginal(X, Y, variance, lengthscale, noise)


def _jax_modules():
    import jax

    # Match GPy float64 reference; default JAX is float32.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import grad, jit

    return jax, jnp, grad, jit


def jax_rbf_K(X, variance, lengthscale):
    """RBF Gram with isotropic lengthscale (JAX)."""
    _, jnp, _, _ = _jax_modules()
    X = jnp.asarray(X, dtype=jnp.float64)
    r2 = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    r2 = r2 / (lengthscale**2)
    return variance * jnp.exp(-0.5 * r2)


def jax_condition(X, Y, variance, lengthscale, noise, jitter=1e-8):
    """Condition once: return LML and an explicit ConditionedState-like dict.

    State keys: ``woodbury_chol``, ``alpha``, ``K``, ``noise``.
    """
    _, jnp, _, _ = _jax_modules()
    X = jnp.asarray(X, dtype=jnp.float64)
    Y = jnp.asarray(Y, dtype=jnp.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    K = jax_rbf_K(X, variance, lengthscale)
    Ky = K + jnp.eye(K.shape[0], dtype=K.dtype) * (noise + jitter)
    L = jnp.linalg.cholesky(Ky)
    # Solve L L^T alpha = Y
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y))
    n, d = Y.shape
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    lml = 0.5 * (-n * d * LOG_2_PI - d * logdet - jnp.sum(alpha * Y))
    state = {
        "woodbury_chol": L,
        "alpha": alpha,
        "K": K,
        "noise": noise + jitter,
        "X": X,
        "variance": variance,
        "lengthscale": lengthscale,
    }
    return lml, state


def jax_predict(state, Xnew):
    """Repeated prediction from conditioned state (no re-factorisation)."""
    _, jnp, _, _ = _jax_modules()
    X = state["X"]
    Xnew = jnp.asarray(Xnew, dtype=jnp.float64)
    L = state["woodbury_chol"]
    alpha = state["alpha"]
    variance = state["variance"]
    lengthscale = state["lengthscale"]
    # Cross-covariance
    r2 = jnp.sum((Xnew[:, None, :] - X[None, :, :]) ** 2, axis=-1) / (lengthscale**2)
    Kx = variance * jnp.exp(-0.5 * r2)
    mean = Kx @ alpha
    v = jnp.linalg.solve(L, Kx.T)
    var = variance - jnp.sum(v * v, axis=0)
    return mean, var[:, None]


def jax_lml_scalar(params, X, Y, jitter=1e-8):
    """Scalar LML for autodiff: params = (variance, lengthscale, noise)."""
    variance, lengthscale, noise = params
    lml, _ = jax_condition(X, Y, variance, lengthscale, noise, jitter=jitter)
    return lml


def run_assessment(seed=0):
    """Compare NumPy GPy ops vs JAX; probe grad/jit; return a result dict."""
    jax, jnp, grad, jit = _jax_modules()

    rng = np.random.RandomState(seed)
    X = rng.randn(20, 1)
    Y = np.sin(X) + 0.1 * rng.randn(20, 1)
    variance, lengthscale, noise = 1.1, 0.8, 0.05

    lml_np, extras_np = numpy_rbf_lml(X, Y, variance, lengthscale, noise)
    lml_jax, state = jax_condition(X, Y, variance, lengthscale, noise)
    lml_jax = float(lml_jax)

    # Parameter gradients via jax.grad on (variance, lengthscale, noise)
    g = grad(lambda p: jax_lml_scalar(p, X, Y))(jnp.array([variance, lengthscale, noise], dtype=jnp.float64))
    g = np.asarray(g, dtype=float)

    # Finite-difference check on JAX objective
    eps = 1e-5
    fd = []
    base = np.array([variance, lengthscale, noise], dtype=float)
    for i in range(3):
        p_hi = base.copy()
        p_lo = base.copy()
        p_hi[i] += eps
        p_lo[i] -= eps
        hi = float(jax_lml_scalar(tuple(p_hi), X, Y))
        lo = float(jax_lml_scalar(tuple(p_lo), X, Y))
        fd.append((hi - lo) / (2 * eps))
    fd = np.asarray(fd)

    # jit of condition LML
    @jit
    def jitted_lml(X_, Y_, v, ls, n):
        lml, _st = jax_condition(X_, Y_, v, ls, n)
        return lml

    lml_jit = float(jitted_lml(X, Y, variance, lengthscale, noise))

    # Predict reuse
    Xnew = rng.randn(5, 1)
    mean, var = jax_predict(state, Xnew)

    return {
        "lml_numpy": float(lml_np),
        "lml_jax": lml_jax,
        "lml_jit": lml_jit,
        "lml_abs_err": abs(lml_jax - float(lml_np)),
        "grad_jax": g,
        "grad_fd": fd,
        "grad_abs_err": np.max(np.abs(g - fd)),
        "predict_mean_finite": bool(np.all(np.isfinite(mean))),
        "predict_var_finite": bool(np.all(np.isfinite(var))),
        "predict_var_nonneg": bool(np.all(var >= -1e-10)),
        "chol_shape": tuple(np.asarray(state["woodbury_chol"]).shape),
    }


def main():
    try:
        _jax_modules()
    except ImportError as err:
        print("SKIP: jax not installed (%s)" % err)
        return 0
    results = run_assessment()
    for k, v in results.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
