"""Compare the grid / CCD designs with a dense brute-force quadrature over the hyperparameters."""
import numpy as np
import GPy
from inla_prototype import INLAHyperparameters, _logsumexp_weights

rng = np.random.default_rng(1)
X = np.sort(rng.uniform(-3, 3, (25, 1)), axis=0)
Y = (rng.uniform(size=(25, 1)) < 1 / (1 + np.exp(-3 * np.sin(1.5 * X)))).astype(float)
k = GPy.kern.RBF(1)
k.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
k.variance.set_prior(GPy.priors.LogGaussian(1.0, 1.0), warning=False)
m = GPy.core.GP(X, Y, kernel=k, likelihood=GPy.likelihoods.Bernoulli(),
                inference_method=GPy.inference.latent_function_inference.Laplace())
Xs = np.linspace(-3.5, 3.5, 8)[:, None]

res = {}
for design in ("grid", "ccd"):
    inla = INLAHyperparameters(m, design=design).fit()
    res[design] = inla.predict_noiseless(Xs) + (len(inla.weights),)

# brute force: dense 41x41 grid in u-space spanning +-5 sd
ref = INLAHyperparameters(m, design="grid").fit()
span = np.linspace(-5, 5, 41)
Z = np.array([[a, b] for a in span for b in span])
U = ref.u_star + Z @ ref.L.T
lp = np.array([ref._log_post(u) for u in U])
ok = np.isfinite(lp)
w = _logsumexp_weights(lp[ok])
mus, vs = [], []
for u in U[ok]:
    m.optimizer_array = u
    mu, v = m.predict_noiseless(Xs); mus.append(mu); vs.append(v)
mus, vs = np.array(mus), np.array(vs)
W = w[:, None, None]
mu_b = (W * mus).sum(0); var_b = (W * (vs + mus**2)).sum(0) - mu_b**2
for d, (mu, var, n) in res.items():
    print(f"{d:5s} n={n:3d} max|dmean|={np.abs(mu-mu_b).max():.3f} max|dvar|={np.abs(var-var_b).max():.3f}")
print(f"dense n={ok.sum()}  mean[0:3]={mu_b.ravel()[:3].round(3)} var[0:3]={var_b.ravel()[:3].round(3)}")
