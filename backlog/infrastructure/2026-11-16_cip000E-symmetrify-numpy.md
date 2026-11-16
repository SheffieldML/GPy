---
id: 2026-11-16_cip000E-symmetrify-numpy
title: CIP-000E — decide symmetrify default (blocked NumPy vs Cython)
status: Completed
priority: Medium
created: "2026-11-16"
last_updated: "2026-11-16"
category: infrastructure
related_cips:
- "000E"
owner: Neil Lawrence
dependencies:
- 2026-11-16_cip000E-triang-maps-numpy
tags:
- backlog
- cip000E
- cython
- symmetrify
- benchmarks
---

# Task: CIP-000E `symmetrify` NumPy/BLAS decision

## Description

Decide blocked NumPy vs Cython for in-place `symmetrify`.

## Acceptance Criteria

- [x] `_symmetrify_blocked` implemented (chunked triangular copy)
- [x] Parity vs `_symmetrify_numpy` (upper/lower)
- [x] **Decision:** default to blocked NumPy; accept ~2–3× vs former Cython
      at N=1000–3000 because packaging simplification outweighs ms-scale cost
      after O(N³) factorisations; unused `cholupdate` removed with `.pyx`
- [x] `linalg.symmetrify` calls `_symmetrify_blocked`

## Related

- CIP: 000E
- Files: `GPy/util/linalg.py`

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept.

### 2026-11-16 (later)

Completed: blocked default; Cython `linalg_cython` deleted in Phase 2.
CIP exception recorded: Cython was faster in harness, but absolute times are
small and Phase 2 packaging win takes priority.
