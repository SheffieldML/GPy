---
id: 2026-11-16_cip000E-triang-maps-numpy
title: CIP-000E — replace flat/triang Cholesky maps with tril_indices NumPy
status: Ready
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

Today `GPy.util.choleskies` still wires **Cython** for
`flat_to_triang` / `triang_to_flat` when the extension builds, even though
Cholesky **backprop** already defaults to Murray blocked.

CIP table: Python loop fallbacks are **80–90×** slower; a `tril_indices`
NumPy rewrite is **~1.1×** vs Cython (effectively tied). Goal under Option C:
make the `tril_indices` form the default (packaging/simplicity win even at
parity), keep Cython importable for soak, and only keep Cython as default if
harness shows a clear regression on realistic (M, D) shapes.

### Anti-patterns (rejected)

- Declaring “done” because backprop flipped while maps still force Cython
- Reintroducing the pure double loops (`_flat_to_triang_pure` /
  `_triang_to_flat_pure`) as the production default — those are the 80–90×
  baselines
- Changing packing order / layout without a compatibility test against
  stored flattened Cholesky parameters

## Acceptance Criteria

- [ ] Implement (or extract) `flat_to_triang` / `triang_to_flat` via
      `np.tril_indices` (or equivalent vectorised scatter/gather) as named
      pure helpers with explicit `(Nflat, D)` / `(D, M, M)` contracts
- [ ] Parity vs Cython and vs current pure loops on random packs; include
      M edge cases (1, 2, large) and D&gt;1
- [ ] Harness under `benchmarks/cip000E/` for maps at M≈200 (CIP size) and
      at least one larger M; report ×cython
- [ ] Flip module-level `flat_to_triang` / `triang_to_flat` defaults to the
      NumPy form when ×cython ≲ 1.2 (tie band) **or** document why Cython
      remains default with numbers attached
- [ ] Call sites that round-trip flat ↔ triang (variational /
      Cholesky-parameterised models) still pass

## Implementation Notes

- Layout must match existing Cython packing (lower triangle, column/stack
  convention used by GPy). Diff against `choleskies_cython.pyx` before
  flipping.
- This is a parameter-transform helper (CIP-000F moderate relevance): keep
  functions pure; no model objects.
- Prefer one shared index cache pattern if D is large and M fixed, but
  measure — do not micro-optimise blindly.

## Related

- CIP: 000E table row `flat_to_triang` / `triang_to_flat`
- Files: `GPy/util/choleskies.py`, `choleskies_cython.pyx`
- Sibling: Cholesky backprop already on Murray default

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept. Ordered after coregionalize.
