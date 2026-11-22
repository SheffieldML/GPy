# CIP-000F Stage 2N — numerical interface sketch

Date: 2026-11-21  
Demo: [`../experiments/exact_gp_conditioned_numpy.py`](../experiments/exact_gp_conditioned_numpy.py)

## Scope

Conceptual boundaries only. **No** GPyTorch-scale `LinearOperator` hierarchy
in production GPy 1. Dense Cholesky remains the reference implementation.

## Protocols (conceptual)

### CovarianceOperator

- Evaluate `K(X, X2)` and `Kdiag(X)`
- Apply covariance to a vector/matrix (`K @ v`)
- Solve linear systems / approximate log-det without assuming every `K` is
  materialised (future); dense Cholesky today

Stage 2 `rbf_ops.K` / `stationary_ops` are the dense reference.

### ConditionedState

Explicit factorisations and reusable posterior artefacts for repeated
prediction (GPJax-style). For exact GP:

| Field | Role |
|---|---|
| `woodbury_chol` | Cholesky of `K + noise + jitter` |
| `alpha` | `K_y^{-1} Y` |
| `X`, kernel params | Needed to form cross-covariance |

Demo: `condition_rbf` once → `predict_from_state` many times. Parity vs
`GPRegression.predict(..., include_likelihood=False)`: mean abs err ~1e-15,
var abs err 0 (seed 1).

Stage 2T JAX experiment returns the same shape of state (feeds
`jax_predict`).

### ParameterSpec

Distinguish:

- **Static structure** — input dim, ARD flag, likelihood family
- **Free numerical values** — variance, lengthscale, noise (plain arrays)
- **Transforms / constraints** — still owned by paramz / Stage 4

Stage 2 already unwraps `Param` once at the method boundary; ParameterSpec
names that split for CIP-0010 / GPy 2.

## Implication for Stage 3 caching

Observer `Cache_this` / `ObsAr` largely compensate for a missing explicit
posterior state. Prefer:

1. **Posterior-scoped** caches attached to a ConditionedState (ideal)
2. **Version-counter interim** on the mutable model if benchmarks require
   keeping GPy 1's decorator cache shape

Stage 3 should ship the version-counter interim without pretending it is the
long-term abstraction; ConditionedState remains the target for the successor.

## Linkage

- CIP-000F Stage 2N acceptance: met (minimal working experiment).
- CIP-0010 Workstreams G / H: cite this note + Stage 2T barrier table.
