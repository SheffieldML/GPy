---
id: 2026-11-16_cip000E-triang-maps-numpy
title: CIP-000E — replace flat/triang Cholesky maps with tril_indices NumPy
status: Completed
priority: Medium
created: "2026-11-16"
last_updated: "2026-11-16"
category: infrastructure
related_cips:
- "000E"
- "000F"
owner: Neil Lawrence
dependencies:
- 2026-11-16_cip000E-coregionalize-reduce-numpy
tags:
- backlog
- cip000E
- cython
- cholesky
- triang
- benchmarks
---

# Task: CIP-000E `flat_to_triang` / `triang_to_flat` NumPy defaults

## Description

Replace Cython flat/triang maps with `np.tril_indices` scatter/gather.

## Acceptance Criteria

- [x] `_flat_to_triang_idx` / `_triang_to_flat_idx` pure helpers
- [x] Parity vs pure loops; defaults wired to idx forms
- [x] Round-trip test in `GPy/testing/test_cython.py`
- [x] Cython path removed in Phase 2 (packaging win; absolute times sub-ms)

## Related

- CIP: 000E
- Files: `GPy/util/choleskies.py`

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept. Ordered after coregionalize.

### 2026-11-16 (later)

Completed: module defaults are `_flat_to_triang_idx` / `_triang_to_flat_idx`.
Harness showed Cython still faster in absolute terms but maps are sub-ms;
Phase 2 deletion preferred for packaging.
