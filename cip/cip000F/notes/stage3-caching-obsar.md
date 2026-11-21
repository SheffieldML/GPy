# CIP-000F Stage 3 — Cache_this / ObsAr

**Date:** 2026-11-21  
**Status:** Implemented on branch `cip000F-stage3-caching-obsar`

## What shipped

1. **GPy-owned** [`GPy/core/parameterization/caching.py`](../../../GPy/core/parameterization/caching.py)
   - Same `Cache_this` / `Cacher` / `FunctionCache` surface as paramz
   - Plain `ndarray` inputs are cacheable (no `ObsAr` required)
   - Cache ids include owner `_cache_version` (works with `ignore_args=(0,)`)
   - Observable inputs still get observer hooks when present

2. **Version bumps**
   - `Parameterized.bump_cache_version` / default `parameters_changed`
   - `Model.bump_cache_version`
   - `GP.parameters_changed` and `GP.set_XY` bump model + kernel

3. **ObsAr → ndarray for model data**
   - `GP._as_data_array` stores X/Y as C-contiguous `float64` ndarrays
   - `ObsAr` kept for EP and pickle via [`observable_array.py`](../../../GPy/core/parameterization/observable_array.py)
   - `sys.modules['GPy.core.parameterization.observable_array']` registered (Accept pickle policy; paramz path still valid while paramz is installed)

## Benchmarks (post-change)

From `cip/cip000F/experiments/stage3_cache_benchmarks.py` on this machine (2026-11-21):

| model | n | mean_s (ll+predict) | std_s |
|-------|---|---------------------|-------|
| GPRegression | 80 | 0.0001 | ~0 |
| SparseGPRegression | 200 | 0.0001 | ~0 |
| BayesianGPLVM | 60 | 0.0001 | ~0 |

Caches that still pay (K, psi, slice helpers) were **kept** with version-counter invalidation rather than removed. No decorator was dropped solely on these timings; Stage 2N’s condition-once/predict-many remains the preferred long-term shape for a successor CIP.

## Relation to Stage 2N

Version-counter interim matches the Stage 2N recommendation: do not put a GPyTorch-scale operator hierarchy in GPy 1; keep explicit invalidation until a posterior-scoped `ConditionedState` can own reuse.
