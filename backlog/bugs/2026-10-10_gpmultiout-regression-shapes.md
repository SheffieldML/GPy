---
id: 2026-10-10_gpmultiout-regression-shapes
title: Fix GPMultioutRegression qU_var shape broadcast error
status: Completed
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
- multioutput
- gpmultiout
---

# Task: `GPMultioutRegression` shape broadcast

## Description

[#733](https://github.com/SheffieldML/GPy/issues/733) reports fitting
`GPy.models.GPMultioutRegression` fails with a `qU_var_r` broadcast error when
`num_inducing[1]` exceeds the number of outputs.

## Acceptance Criteria

- [x] Reproduce on current `devel`
- [x] Align `qU_var_r_W` / `qU_var_r_diag` dimensions with the model definition
- [x] Unit test for multi-output fit that previously failed
- [x] Comment / close #733

## Implementation Notes

Root cause: default `num_inducing=(10,10)` sized `qU_var_r_*` at Mr=10 while
`Z_row` only had `D` rows. Fix (#1173): cap Mr at D and size q(U) from actual
`Z` / `Z_row`.

## Related

- Issues: #733 (closed)
- PR: #1173 (merged)

## Progress Updates

### 2026-10-10

Task created from open-issue triage; fixed and closed via #1173.
