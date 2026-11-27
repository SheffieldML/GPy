"""Accuracy of the ccd / grid designs for d >= 4 hyperparameters (CIP-0015).

Reference: self-normalised importance sampling over the hyperparameters in the standardised
z-space, with a heavy-tailed multivariate-t proposal (so tails are covered), on the *same*
target p(u | y) that the wrapper uses. The likelihood is Gaussian with fixed noise so that a
few thousand evaluations are cheap; the wrapper does not care which likelihood is used.
"""

import sys
import time

import numpy as np

import GPy
from GPy.inference.hyperparameters import INLAHyperparameters


def make_model(n_inputs, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, n_inputs))
    # first input matters most, last input is irrelevant
    w = np.linspace(2.5, 0.0, n_inputs)
    Y = np.sin(3 * X @ w[:, None] / max(w.sum(), 1)) + 0.15 * rng.standard_normal((n, 1))
    k = GPy.kern.RBF(n_inputs, ARD=True)
    k.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    k.variance.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    lik = GPy.likelihoods.Gaussian(variance=0.05)
    lik.variance.fix()
    m = GPy.core.GP(X, Y, kernel=k, likelihood=lik)
    return m, rng


def importance_reference(inla, Xs, n_samples, rng, dof=5, scale=1.2):
    d = inla.u_star.size
    g = rng.standard_normal((n_samples, d))
    chi = rng.chisquare(dof, size=(n_samples, 1))
    Z = scale * g / np.sqrt(chi / dof)
    log_q = -0.5 * (dof + d) * np.log1p((Z / scale) ** 2 @ np.ones(d) / dof)  # up to a constant
    U = inla.u_star + Z @ inla.L.T
    log_p = np.array([inla._log_post(u) for u in U])
    ok = np.isfinite(log_p)
    log_w = log_p[ok] - log_q[ok]
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    ess = 1.0 / (w**2).sum()
    mus, vs, ths = [], [], []
    m = inla.model
    for u in U[ok]:
        m.optimizer_array = u
        a, b = m.predict_noiseless(Xs)
        mus.append(a)
        vs.append(b)
        ths.append(np.array(m.unfixed_param_array, dtype=float))
    m.optimizer_array = inla.u_star
    mus, vs, ths = np.array(mus), np.array(vs), np.array(ths)
    W = w[:, None, None]
    mu = (W * mus).sum(0)
    var = (W * (vs + mus**2)).sum(0) - mu**2
    th_mean = (w[:, None] * ths).sum(0)
    th_sd = np.sqrt((w[:, None] * (ths - th_mean) ** 2).sum(0))
    return mu, var, th_mean, th_sd, ess


def run(n_inputs, n_samples=20000, designs=("ccd", "grid", "imp250", "imp1000", "imp4000")):
    m, rng = make_model(n_inputs)
    Xs = rng.uniform(-1, 1, (30, n_inputs))
    inla = INLAHyperparameters(m, design="ccd").fit()
    d = inla.u_star.size
    t = time.time()
    mu_r, var_r, th_m_r, th_sd_r, ess = importance_reference(inla, Xs, n_samples, rng)
    print(f"\n=== d={d} (reference: {n_samples} IS draws, ESS={ess:.0f}, {time.time() - t:.0f}s) ===")
    mu_map, var_map = m.predict_noiseless(Xs)
    scale = np.sqrt(var_r)
    names = list(inla.hyperparameter_summary())
    rows = {"ref": (th_m_r, th_sd_r)}
    print(
        f"MAP plug-in : max|dmean|/sd={np.abs(mu_map - mu_r).max() / scale.mean():.3f}  "
        f"var ratio (MAP/ref) mean={np.mean(var_map / var_r):.3f}"
    )
    for design in designs:
        t = time.time()
        if design.startswith("imp"):
            i = INLAHyperparameters(m, design="importance", n_samples=int(design[3:])).fit()
        else:
            i = INLAHyperparameters(m, design=design).fit()
        mu, var = i.predict_noiseless(Xs)
        s = i.hyperparameter_summary()
        th_m = np.array([v[0] for v in s.values()])
        th_sd = np.array([v[1] for v in s.values()])
        rows[design] = (th_m, th_sd)
        print(
            f"{design:7s} n={len(i.weights):5d} ess={i.effective_sample_size:6.0f} ({time.time() - t:4.0f}s)  "
            f"max|dmean|/sd={np.abs(mu - mu_r).max() / scale.mean():.3f}  "
            f"var ratio mean={np.mean(var / var_r):.3f} range=({(var / var_r).min():.2f}, {(var / var_r).max():.2f})  "
            f"theta mean rel.err={np.abs(th_m / th_m_r - 1).max():.3f}  theta sd rel.err={np.abs(th_sd / th_sd_r - 1).max():.3f}"
        )

    print("hyperparameter marginals (mean +- sd, constrained scale):")
    for j, name in enumerate(names):
        print(f"  {name:28s}" + "  ".join(f"{k}: {v[0][j]:6.3f}+-{v[1][j]:6.3f}" for k, v in rows.items()))


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [3, 4]
    for k in dims:
        # d = k + 1 hyperparameters (k ARD lengthscales + variance); the full grid is only run for small d
        run(k)
