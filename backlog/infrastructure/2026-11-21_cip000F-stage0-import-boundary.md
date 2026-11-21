---
id: 2026-11-21_cip000F-stage0-import-boundary
title: CIP-000F Stage 0 — single paramz import boundary + Ruff TID251
status: Ready
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

- [ ] `core/parameterization/caching.py` re-exports `Cache_this` / `Cacher`
- [ ] Domains (and any other missing symbols) re-exported from the façade
- [ ] Direct `paramz` imports remain only in the allowed packages
      (target: ≤ ~8 modules)
- [ ] Ruff `TID251` enabled in the same PR
- [ ] Full `GPy/testing` suite green; public model API unchanged

## Implementation Notes

Worktree/branch: `cip000F-paramz-reduction`. One PR for the whole stage.

## Related

- CIP: 000F
- Discussion: https://github.com/SheffieldML/GPy/discussions/1293

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.
