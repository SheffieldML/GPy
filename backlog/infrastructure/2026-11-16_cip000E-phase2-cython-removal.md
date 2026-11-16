---
id: 2026-11-16_cip000E-phase2-cython-removal
title: CIP-000E Phase 2 — remove Cython build surface after hybrid soak
status: Proposed
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

After Option C hybrid soak, remove the five Cython extension modules and the
packaging/CI cost they impose (toward Option A).

**Do not start this task** until Phase 1 defaults are NumPy/BLAS (or an
explicit CIP-recorded exception such as `symmetrify`), Linux multicore
harness rows exist for the flipped hotspots, and a short soak on
`cip000E-numpy-blas-primitives` (or merged main) shows no correctness
regressions.

## Acceptance Criteria

- [ ] Phase 1 dependency tasks Completed (or CIP Accept notes exceptions)
- [ ] Linux multicore + threading evidence attached for Cholesky Murray and
      each flipped hotspot
- [ ] Delete or stop building: `stationary_cython`, `choleskies_cython`,
      `linalg_cython`, `coregionalize_cython`, `state_space_cython` (and
      `stationary_utils.c` if only used there)
- [ ] Remove Extension list / OpenMP flags from `build_extension.py` (or
      delete the hook once CIP-000C backend no longer needs it)
- [ ] `[cython] working` / `use_cython` become no-ops or deprecated with a
      documented window
- [ ] CI: drop Cython compile from PR matrix where CIP-000A allows; wheels
      become pure-Python (or abi3) as applicable
- [ ] Update CIP-0008 / 000A / 000C cross-references
- [ ] `pytest GPy/testing` green without Cython extensions present
- [ ] CIP-000E Phase 2 checklist item marked done; prepare Close / compress

## Implementation Notes

- State-space: Phase 0 already fixed import; CIP evidence says Cython ≈
  Python. Prefer deletion over keeping a dead accelerator. Option D
  (Numba/JAX scan) is a **separate** follow-on, not a Phase 2 blocker.
- Coordinate with CIP-000C so uv/Poetry build-backend work does not invest
  further in Cython hooks.
- Keep numerical primitives (`cholesky_vjp_blocked`, stationary helpers,
  etc.) — only the `.pyx` / compile path goes away.

## Related

- CIP: 000E Accept Option C → Phase 2 toward A
- Packaging CIPs: 0008, 000A, 000C

## Progress Updates

### 2026-11-16

Task created **Proposed** (not Ready) until Phase 1 soak evidence exists.
Promote to Ready only after dependency tasks complete.
