---
id: 2026-10-10_heteroscedastic-set-xy
title: Fix set_XY for heteroscedastic Gaussian regression
status: Proposed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips: []
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
list-aware `set_XY` + normalizer (#1149).

Open-issue triage (2026-10-10): backlog bug; pair mentally with the
heteroscedastic + mean_function feature task.

## Acceptance Criteria

- [ ] Reproduce #959 on current `devel`
- [ ] `set_XY` updates `X`, `Y`, and heteroscedastic noise metadata consistently
- [ ] Regression test for train → `set_XY` → predict
- [ ] Comment / close #959

## Implementation Notes

- Inspect how heteroscedastic likelihood stores per-point noise and whether
  `Y_metadata` is refreshed on `set_XY`.
- Coordinate with the mean_function feature task if the wrapper needs shared
  API cleanup.

## Related

- Issues: #959
- Related backlog: `2026-10-10_heteroscedastic-mean-function`
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.
