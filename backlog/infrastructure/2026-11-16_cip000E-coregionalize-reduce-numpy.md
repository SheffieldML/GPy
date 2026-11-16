---
id: 2026-11-16_cip000E-coregionalize-reduce-numpy
title: CIP-000E — replace coregionalize gradient_reduce with one-hot BLAS form
status: Ready
priority: High
created: "2026-11-16"
last_updated: "2026-11-16"
category: infrastructure
related_cips:
- "000E"
- "000F"
owner: Neil Lawrence
dependencies:
- 2026-11-16_cip000E-stationary-grads-numpy
tags:
- backlog
- cip000E
- cython
- coregionalize
- benchmarks
---

# Task: CIP-000E coregionalize `gradient_reduce` NumPy replacement

## Description

Replace `coregionalize_cython.gradient_reduce` as the wired default with the
CIP-cited form: **one-hot / indicator reduction** equivalent to
`O.T @ dL_dK @ O` (early bench **0.6×** vs Cython at N=5000, D=5).

`coregionalize` `K` already ties Cython at ~1× with the NumPy indexer
fallback — **do not spend Phase 1 effort “optimising” `K`** unless a new
harness shows a real gap. Focus is `gradient_reduce`, which is the
double-index reduction used in `update_gradients_full`.

### Anti-patterns (rejected)

- Leaving `_gradient_reduce_numpy`’s double Python loop as the new default
  (CIP table: **2.8× slower** than Cython)
- A “clever” Cython-shaped rewrite in Python that still walks indices in
  interpreted loops
- Changing `K` / `Kdiag` APIs or B-matrix semantics as a side quest

## Acceptance Criteria

- [ ] Pure primitive, e.g.
      `gradient_reduce(dL_dK, index, index2, output_dim) -> (D, D)`,
      array inputs only; no `Coregionalize` instance required
- [ ] Parity vs Cython and vs `_gradient_reduce_numpy` at representative
      index patterns (repeated outputs, asymmetric X/X2, D&gt;2)
- [ ] Harness entry under `benchmarks/cip000E/` (N, D, index entropy,
      wall time, ×cython, peak memory, BLAS/threads)
- [ ] ×cython **&lt; 1** (or documented tie within noise) on CIP-like size
      (N≈5000, D≈5) before flipping the default in
      `Coregionalize.update_gradients_full`
- [ ] Cython path remains importable for hybrid soak
- [ ] Integration: existing coregionalized regression / multioutput tests green

## Implementation Notes

- Preferred construction: build sparse or dense indicator `O` (N×D) with
  one-hot rows from `index`, then `O.T @ dL_dK @ O` (or an equivalent
  `np.add.at` / bincount reduction that matches that algebra and wins the
  harness). Prove equivalence to the nested-loop reduce.
- Watch dtype: Cython uses `int64` indices; keep that contract at the
  wrapper boundary.
- `K_symmetric` / `K_asymmetric` Cython can stay as-is for this task.

## Related

- CIP: 000E table row `coregionalize gradient_reduce`
- Files: `GPy/kern/src/coregionalize.py`, `coregionalize_cython.pyx`
- Docs touchpoint: coregionalized regression examples (no behaviour change
  expected)

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept. Ordered after stationary grads
per Accept decisions.
