---
id: 2026-11-27_exact-gaussian-jitter
title: Make the exact Gaussian inference noise jitter configurable
status: Completed
priority: Medium
created: "2026-11-27"
last_updated: "2026-11-27"
category: features
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- inference
- numerics
---

# Task: Configurable jitter in ExactGaussianInference

## Description

`ExactGaussianInference` adds 1e-8 to the noise variance before factorising `K + noise`. The
model GPy evaluates therefore has noise variance `variance + 1e-8`, not the stated `variance`.
The relative change is 1e-5 at variance 1e-3 and 1e-4 at variance 1e-4. A 60-digit reference
evaluation, and cross-framework comparison against GPflow, GPJax and GPyTorch in gp-scuderia,
confirmed this is the whole of the gap in the log likelihood and its gradients at small noise.

Expose the constant as `ExactGaussianInference(jitter=1e-8)`. The default is unchanged, so
behaviour is identical. Making it an explicit option documents what the term does and lets us
warn about, or change, the default in future.

## Acceptance Criteria

- [x] `ExactGaussianInference(jitter=...)` passes the value to `exact_gp_ops.log_marginal_and_gradients`.
- [x] Default behaviour is numerically unchanged.
- [x] `jitter=0` matches an exact evaluation of the stated model.
- [x] Serialization round-trips a non-default jitter and omits the default, so older readers still load files.
- [x] Docstring states that jitter is added to the noise variance.

## Implementation Notes

Not changed here: `exact_studentt_inference` (hard-coded 1e-8), `gaussian_grid_inference`, and the
sparse approximations (`const_jitter`). `EP` inherits `ExactGaussianInference`; the implementation
reads the attribute with a default so subclasses that skip `__init__` keep working.

## Related

- gp-scuderia `benchmarks/gradient_agreement.yaml` case `small_noise` and `benchmarks/conditioning.yaml` case `noise_1e-4`
  could be removed once the adapter passes `jitter=0`.

## Progress Updates

### 2026-11-27
Created and implemented with tests in `GPy/testing/test_exact_gaussian_jitter.py`.
