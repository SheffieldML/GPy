---
id: 2026-11-16_cip000E-coregionalize-reduce-numpy
title: CIP-000E — replace coregionalize gradient_reduce with one-hot BLAS form
status: Completed
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

- [x] Pure primitive `coregionalize_ops.gradient_reduce(dL_dK, index, index2, output_dim) -> (D, D)`
- [x] Parity vs Cython and nested-loop oracle (`TestCoregionalizeReduce`)
- [x] Harness `benchmarks/cip000E/run_coregionalize_reduce.py`
- [x] macOS/Accelerate N=5000 D=5: one-hot **~0.33×** Cython; defaults flipped
- [x] Cython path retained for soak; `K` left on existing NumPy/Cython dispatch
- [x] `pytest … -k coreg` green

## Implementation Notes

- GPy stores `out[j, i] = sum` over `index==i`, `index2==j`, so the primitive
  returns `(O.T @ dL @ O2).T` to match Cython/loop layout (not the untransposed
  `O.T @ dL @ O2`).
- `np.add.at` matched numerically but was **slower** than Cython on this host;
  one-hot GEMM is the winning form.

## Related

- CIP: 000E table row `coregionalize gradient_reduce`
- Files: `GPy/kern/src/coregionalize.py`, `coregionalize_ops.py`, `coregionalize_cython.pyx`

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept.

### 2026-11-16 (implementation)

Landed one-hot BLAS default on `cip000E-numpy-blas-primitives`.
