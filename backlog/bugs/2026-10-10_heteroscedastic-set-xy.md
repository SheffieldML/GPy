---
id: 2026-10-10_heteroscedastic-set-xy
title: Fix set_XY for heteroscedastic Gaussian regression
status: Done
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- heteroscedastic
- set-xy
---

# Task: Heteroscedastic `set_XY`

## Description

[#959](https://github.com/SheffieldML/GPy/issues/959): after training a
heteroscedastic Gaussian regression model, updating observations via `set_XY`
for successive forecasting fails. Distinct from coregionalized
list-aware `set_XY` + normalizer (#1149). Same failure as [#858](https://github.com/SheffieldML/GPy/issues/858).

Open-issue triage (2026-10-10): backlog bug; pair mentally with the
heteroscedastic + mean_function feature task.

## Acceptance Criteria

- [x] Reproduce #959 on current `devel`
- [x] `set_XY` updates `X`, `Y`, and heteroscedastic noise metadata consistently
- [x] Regression test for train → `set_XY` → predict
- [x] Comment / close #959 (and #858) via #1169

## Implementation Notes

- Per-point `het_Gauss.variance` and `Y_metadata['output_index']` stayed at the
  old `N` after `set_XY`, so exact inference broadcast-failed
  `(new_N,) vs (old_N,)`.
- `GP.set_XY` now accepts optional `Y_metadata` and calls
  `_sync_heteroscedastic_noise`; `HeteroscedasticGaussian.resize_for_data`
  relinks the variance `Param` when the row count changes.
- New noise rows default to the mean of the previous variances.

## Related

- Issues: #959, #858
- Related backlog: `2026-10-10_heteroscedastic-mean-function`
- Branch: `fix/959-heteroscedastic-set-xy`
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

### 2026-10-10 (execution)

Reproduced; fix + regression test on `fix/959-heteroscedastic-set-xy`.
