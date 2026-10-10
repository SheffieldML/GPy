---
id: 2026-10-10_gpmultiout-regression-shapes
title: Fix GPMultioutRegression qU_var shape broadcast error
status: In Progress
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
`GPy.models.GPMultioutRegression` fails with:

```text
qU_var_r = tdot(self.qU_var_r_W) + np.diag(self.qU_var_r_diag)
ValueError: operands could not be broadcast together with shapes (4,4) (10,10)
```

## Acceptance Criteria

- [x] Reproduce on current `devel`
- [x] Align `qU_var_r_W` / `qU_var_r_diag` dimensions with the model definition
- [x] Unit test for multi-output fit that previously failed
- [ ] Comment / close #733

## Implementation Notes

- Root cause: default `num_inducing=(10,10)` sized `qU_var_r_*` at Mr=10 while
  `Z_row` is sampled from `X_row` with only `D` rows (outputs), so effective Mr
  became `D`. Also `qU_mean = np.zeros(num_inducing)` built a square `(Mc, Mc)`.
- Fix: cap Mr at D (warn), size all q(U) factors from actual `Z` / `Z_row`.

## Related

- Issues: #733
- Branch: `fix/733-gpmultiout-shapes`

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

### 2026-10-10 (execution)

Reproduced; fix + regression test on `fix/733-gpmultiout-shapes`.
