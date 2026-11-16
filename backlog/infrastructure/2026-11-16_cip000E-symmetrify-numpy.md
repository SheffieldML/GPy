---
id: 2026-11-16_cip000E-symmetrify-numpy
title: CIP-000E — decide symmetrify default (blocked NumPy vs Cython)
status: Ready
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

`GPy.util.linalg.symmetrify` still prefers `linalg_cython.symmetrify` when
built. CIP table: NumPy fallback is **7–9×** slower; a **blocked transpose
copy** rewrite was **2–3×** at N=1000 and **~0.6×** at N=3000 vs Cython.

This is the one hotspot where Cython may still win at small N and where
absolute times are milliseconds after an O(N³) factorisation. Decision must
be evidence-based, not aesthetic.

### Anti-patterns (rejected)

- Deleting Cython `symmetrify` without a blocked NumPy candidate that the
  harness actually timed
- Promoting `_symmetrify_numpy` (simple `triu_indices` copy) as “good enough”
  without measuring — that is the 7–9× baseline
- Blocking Phase 2 forever on this helper if larger hotspots are already
  NumPy-default and `symmetrify` is proven negligible in profiles

## Acceptance Criteria

- [ ] Implement a **blocked** in-place symmetrify candidate (chunked
      triangular copy / transpose tiles), as a pure function on an `ndarray`
- [ ] Parity vs Cython (upper and lower modes) including non-contiguous /
      Fortran-order inputs if Cython supports them today
- [ ] Harness under `benchmarks/cip000E/` for N ∈ {1000, 3000} (and one
      small N); report ×cython and whether the call is measurable in an SVGP
      step profile
- [ ] **Decision recorded in CIP-000E:**
  - Flip default to blocked NumPy if ×cython &lt; 1 at relevant N **or**
    profile share is negligible and packaging simplification wins; **or**
  - Keep Cython as default for `symmetrify` through hybrid soak with
    explicit rationale and numbers
- [ ] Unused `cholupdate` in `linalg_cython.pyx`: mark dead or remove in the
      same PR if touch the module (hygiene item from CIP motivation)

## Implementation Notes

- In-place contract must remain (`symmetrify` mutates `A`).
- If blocked NumPy wins only for large N, consider a size threshold — but
  justify with harness data, do not guess.
- Lowest leverage of Phase 1; do not let it jump the queue ahead of
  stationary / coregionalize / triang.

## Related

- CIP: 000E table row `symmetrify`; hygiene note on unused `cholupdate`
- Files: `GPy/util/linalg.py`, `linalg_cython.pyx`

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept. Last Phase 1 numerical hotspot
before Phase 2 packaging removal.
