---
id: 2026-11-16_cip000E-phase2-cython-removal
title: CIP-000E Phase 2 — remove Cython build surface after hybrid soak
status: Completed
priority: Medium
created: "2026-11-16"
last_updated: "2026-11-16"
category: infrastructure
related_cips:
- "000E"
- "0008"
- "000A"
- "000C"
owner: Neil Lawrence
dependencies:
- 2026-11-16_cip000E-cholesky-murray-soak
- 2026-11-16_cip000E-stationary-grads-numpy
- 2026-11-16_cip000E-coregionalize-reduce-numpy
- 2026-11-16_cip000E-triang-maps-numpy
- 2026-11-16_cip000E-symmetrify-numpy
tags:
- backlog
- cip000E
- cython
- packaging
- wheels
---

# Task: CIP-000E Phase 2 Cython build-surface removal

## Description

Remove the five Cython extension modules and packaging/CI cost (Option C → A).

## Acceptance Criteria

- [x] Phase 1 defaults NumPy/BLAS (incl. triang + blocked symmetrify)
- [ ] Linux multicore soak (optional follow-up; Mac evidence + packaging win
      justified proceeding)
- [x] Deleted `.pyx` / `stationary_utils.c|.h` / `build_extension.py`
- [x] Removed `[tool.poetry.build]` and Cython from build-system requires
- [x] `[cython] working = False`; runtime `use_*_cython = False`
- [x] CI: pure-Python smoke import; no Cython extension smoke
- [ ] Cross-ref CIP-0008 / 000A / 000C (checklist remains on CIP-000E)
- [x] Tests updated for NumPy path parity without Cython

## Related

- CIP: 000E

## Progress Updates

### 2026-11-16

Task created Proposed.

### 2026-11-16 (later)

Completed on `cip000E-numpy-blas-primitives`: pure-Python package; Cython
surface removed. Linux multicore soak and packaging CIP cross-refs remain
as CIP-000E checklist items.
