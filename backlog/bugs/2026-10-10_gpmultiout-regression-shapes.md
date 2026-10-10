---
id: 2026-10-10_gpmultiout-regression-shapes
title: Fix GPMultioutRegression qU_var shape broadcast error
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

`tdot(self.qU_var_r_W)` scales with output dimensions while
`np.diag(self.qU_var_r_diag)` appears tied to another dimension (e.g. inducing
count). Distinct from coregionalized `set_XY` / normalizer work (#1149) and
FITC multi-output (#821).

Open-issue triage (2026-10-10): backlog bug.

## Acceptance Criteria

- [ ] Reproduce on current `devel` (or document that the model path is gone /
      renamed)
- [ ] Align `qU_var_r_W` / `qU_var_r_diag` dimensions with the model definition
- [ ] Unit test for multi-output fit that previously failed
- [ ] Comment / close #733

## Implementation Notes

- Audit `GPMultioutRegression` / related variational multi-output models for
  unused or half-maintained code paths before investing heavily.
- Prefer a clear deprecation + docs if the model is superseded by
  `GPCoregionalizedRegression` / `MultioutputGP`.

## Related

- Issues: #733
- Open-issue triage: 2026-10-10
- Proposed parent CIP: 0007 (open-issue triage), when accepted

## Progress Updates

### 2026-10-10

Task created from open-issue triage.
