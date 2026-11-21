---
id: 2026-11-21_cip000F-stage0-import-boundary
title: CIP-000F Stage 0 — single paramz import boundary + Ruff TID251
status: Completed
priority: High
created: "2026-11-21"
last_updated: "2026-11-21"
category: infrastructure
related_cips:
- "000F"
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- cip000F
- paramz
- ruff
---

# Task: CIP-000F Stage 0 import boundary

## Description

Route every `paramz` import outside `GPy.core.parameterization` and
`GPy.inference.optimization` through GPy façades. Add Ruff `TID251`
(`banned-api`) so new direct imports cannot regress. No behaviour change;
full test suite unchanged.

## Acceptance Criteria

- [x] `core/parameterization/caching.py` re-exports `Cache_this` / `Cacher`
- [x] Domains (and any other missing symbols) re-exported from the façade
- [x] Direct `paramz` imports remain only in the allowed packages
      (10 modules inside `core/parameterization/` + `inference/optimization/`;
      zero outside)
- [x] Ruff `TID251` enabled in the same PR
- [x] Representative `GPy/testing` suite green; public model API unchanged

## Implementation Notes

Worktree/branch: `cip000F-paramz-reduction`. One PR for the whole stage.

Façade also re-exports `ObsAr`, `ParamzModel`, `paramz_load`,
`ParametersChangedMeta`, and `Pickleable` from
`GPy.core.parameterization` for callers that previously imported those
from `paramz` directly.

## Related

- CIP: 000F
- Discussion: https://github.com/SheffieldML/GPy/discussions/1293

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.

### 2026-11-21 (later)

Stage 0 implemented on `cip000F-paramz-reduction`: façades, import rewrite,
TID251. Ruff clean; `GPy/testing` (excluding plotting/deactivated) —
469 passed, 11 skipped.
