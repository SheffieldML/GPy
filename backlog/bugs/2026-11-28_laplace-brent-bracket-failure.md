---
id: "2026-11-28_laplace-brent-bracket-failure"
title: "Laplace mode finding fails with scipy BracketError during hyperparameter optimisation"
status: "Proposed"
priority: "Medium"
created: "2026-11-28"
last_updated: "2026-11-28"
category: "bugs"
related_cips: ["0015"]
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- laplace
- binomial
- inla
---

# Task: Laplace line search raises BracketError for a Binomial GP

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).

## Description

`Laplace.rasm_mode` (`GPy/inference/latent_function_inference/laplace.py`, line ~220) uses `scipy.optimize.brent(inner_obj, tol=1e-4, maxiter=12)` for the Newton step size. With some hyperparameter values scipy cannot find a bracket and raises `BracketError: The algorithm terminated without finding a valid bracket`. The exception escapes from `parameters_changed`, so it aborts the outer optimiser instead of being treated as a bad point.

Found while writing a Tokyo-rainfall-style example for `cip/cip0015` (see `inla_vs_hmc_poisson.ipynb`). Switching to a Poisson likelihood avoided it, so this has not been seen with Poisson or Bernoulli.

### Reproduction

The model: 366 days, `Binomial` likelihood with `Y_metadata={'trials': ...}` (2 trials, 1 on day 60), `StdPeriodic(period=366)` with the period fixed, and `LogGaussian` priors on variance and lengthscale, `Laplace` inference. The data are simulated:

```python
rng = np.random.default_rng(3)
days = np.arange(1, 367); trials = np.full(366, 2.); trials[59] = 1.
p = 1/(1+np.exp(-(-1.0 + 1.2*np.sin(2*np.pi*days/366 - 1.0)
                  + 0.6*np.sin(4*np.pi*days/366) + 0.4*np.sin(6*np.pi*days/366 + 1))))
Y = rng.binomial(trials.astype(int), p)[:, None].astype(float)
# build GP as above, then:
INLAHyperparameters(m).fit()   # raises BracketError inside m.optimize()
```

The traceback goes `INLAHyperparameters.fit` -> `m.optimize` -> L-BFGS -> `Laplace.rasm_mode` -> `optimize.brent`. The model's log likelihood is finite at the starting values and on a grid of variance (1e-3 to 1e4) and lengthscale (0.1 to 1e3), so the failing point is reached during the optimisation. After the failure the variance sits near 1e-8, a nearly degenerate prior, which suggests the optimiser stepped to an extreme value. We have not yet confirmed which point fails.

## Acceptance Criteria

- [x] A minimal script that reproduces the error and fails on `main`, captured as a test in `GPy/testing` (`test_laplace_line_search.py`)
- [x] The failing hyperparameter values identified, and the cause established (flat objective at the line-search start)
- [ ] `rasm_mode` handles a failed line search without aborting the outer optimiser, for example a bounded fallback step, or a clear signal that the point is infeasible
- [ ] `INLAHyperparameters(m).fit()` succeeds on the reproduction model
- [ ] Existing Laplace and INLA tests still pass

## Implementation Notes

- The other `optimize.brent` call (line ~428) has the same pattern and probably the same weakness.
- Compare the objective at step sizes 0 and 1 first. If it is flat or `nan`, fall back to step 1 or a backtracking line search.
- The related check `new_obj < old_obj` raising "Shouldn't happen" is a second route to the same kind of abort.

## Related

- CIP: 0015
- Notebook: `cip/cip0015/inla_vs_hmc_poisson.ipynb`

## Progress Updates

### 2026-11-28
Task created. Reproduced with the model above. Root cause not yet identified.

### 2026-11-28 (reproduction and cause)
Reproduced with a direct hyperparameter scan (no optimiser needed): the Binomial/`StdPeriodic` model raises `BracketError` whenever the kernel variance is very small (1e-8 always; 1e-6 for some data), for any lengthscale. The scanned values are exactly where the optimiser ends up after the failure. The INLA fit on the original simulation did not fail on its own here, so the optimiser path depends on the data.

Cause: the Newton iteration has already converged when the line search starts, so the objective is flat in the step size. At the failing call `inner_obj` returns the identical value (408.26355383528335) at steps -1, 0, 1 and 2, and `brent` cannot bracket a minimum of a constant function. This is not a non-concave case. The same applies to the `new_obj < old_obj` check, which can fire on round-off.

Tests in `GPy/testing/test_laplace_line_search.py` (variance 1e-8, two lengthscales) fail on `main`. Next: the fallback in `rasm_mode` and the second `brent` call.
