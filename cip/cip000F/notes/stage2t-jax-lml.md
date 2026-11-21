# CIP-000F Stage 2T — JAX exact-GP RBF LML assessment

Date: 2026-11-21  
Experiment: [`../experiments/exact_gp_rbf_lml_jax.py`](../experiments/exact_gp_rbf_lml_jax.py)  
Reference: `GPy.inference.latent_function_inference.exact_gp_ops.rbf_log_marginal`

## Setup

- Out-of-package script; JAX installed only in the local developer/CI optional
  environment. **Not** added to GPy `pyproject.toml` dependencies.
- `jax.config.update("jax_enable_x64", True)` required for float64 parity with
  GPy / NumPy.
- Data: N=20, Q=1, isotropic RBF; variance=1.1, lengthscale=0.8, noise=0.05.

## Results (seed 0)

| Check | Result |
|---|---|
| LML NumPy vs JAX abs error | ~2e-15 |
| LML under `jax.jit` vs eager | match within float64 noise |
| `jax.grad` vs central FD (params) | max abs err ~1e-7 |
| Conditioned predict mean/var finite | yes |
| Predictive variance ≥ 0 | yes |

## Barrier classification

| Class | Finding |
|---|---|
| **Incidental implementation coupling** | GPy `pdinv` / `dpotrs` / `diag.add` are SciPy/LAPACK wrappers; the JAX path uses `jnp.linalg.cholesky` + triangular solves. No paramz / Model coupling in either pure path. |
| **Control-flow or shape limitation** | Returning a Python `dict` conditioned state from a `@jit` function that also computes LML is awkward; the assessment jitted a scalar LML only. Structured state should be a pytree (namedtuple / dataclass) for full `jit` of condition+predict. |
| **Numerical limitation** | Default JAX float32 diverges (~1e-5 LML, ~0.3 on noise gradient). Explicit float64 closes the gap. |
| **Backend-specific optimisation** | Dense Cholesky is fine for N=20; larger N would want backend-specific factorisations / operators (feeds Stage 2N `CovarianceOperator` note). |
| **Fundamental modelling constraint** | None observed for isotropic exact GP LML. |

GPy 1 actionable without behaviour change: keep Stage 2 pure NumPy ops; do not
pull JAX into the runtime. Prefer explicit conditioned payloads (chol, alpha)
for future prediction reuse (Stage 2N).

## Linkage

- CIP-000F Stage 2T acceptance: met for one representative objective.
- CIP-0010 Workstream G (transformation-readiness): cite this note.
