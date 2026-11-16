---
id: 2026-11-16_cip000E-stationary-grads-numpy
title: CIP-000E — replace stationary Cython grads with CIP-cited NumPy/BLAS forms
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
- 2026-11-16_cip000E-cholesky-murray-soak
tags:
- backlog
- cip000E
- cython
- stationary
- kernels
- benchmarks
---

# Task: CIP-000E stationary lengthscale / gradients_X NumPy replacements

## Description

Replace `stationary_cython.lengthscale_grads` and `stationary_cython.grad_X`
as the **wired defaults** under CIP-000E Option C, using the formulations
already identified in the CIP benchmark table — not weaker loop rewrites
that merely “look like NumPy.”

| Op | CIP-cited winning form | Early ×cython (Linux) |
|---|---|---|
| ARD `lengthscale_grads` | **one matmul** over the stacked \((X_{nq}-X2_{mq})\) structure | **0.15×** at N=3000, Q=10 |
| `gradients_X` | **`X * rowsum - tmp @ X2`** (GEMM), not the per-dimension Python loop in `_gradients_X_pure` | **0.28×** at N=3000, Q=10 |

Current pure fallbacks (`_lengthscale_grads_pure`, `_gradients_X_pure`) are
explicitly the *slow* baselines in the CIP table (4.2× / 6.3× vs Cython).
Shipping those as the new default would be a regression. The replacement
must be the fast matrix form, parity-checked against Cython/oracle, then
measured.

### Anti-patterns (rejected)

- Flipping the default to the existing pure Python/`for q` loops without a
  new primitive that beats Cython on the harness
- Broadcasting an \(N \times M \times Q\) temporary as the only strategy
  without recording peak memory (CIP vectorisation warning)
- Coupling the primitive to `Kern` / `paramz` state; Stage 2 wants array in /
  array out, with the kernel method as a thin wrapper

## Acceptance Criteria

- [x] Pure primitives in `GPy/kern/src/stationary_ops.py`:
  - `lengthscale_grads_sqdiff(tmp, X, X2) -> (Q,)`
  - `gradients_X_from_tmp(tmp, X, X2) -> (N, Q)` (GEMM form)
  Both callable with plain `ndarray`s; no `Kern` / `Param` required
- [x] Numerical parity vs Cython and vs pure path
      (`GPy/testing/test_cython.py::TestStationary`)
- [x] Harness `benchmarks/cip000E/run_stationary_grads.py`
- [x] macOS/Accelerate N=M=3000 Q=10: lengthscale **0.056×**, grad_X
      **0.044×** vs Cython — defaults flipped
- [x] Peak tracemalloc recorded in harness (blas allocates more than Cython
      for lengthscale ~516 KiB vs ~1 KiB traced; still far below an
      \(N\times M\times Q\) broadcast). No \(N\times M\times Q\) temporary used.
- [x] `stationary.py` defaults to BLAS helpers; Cython methods retained for soak
- [x] CIP-000F: JAX/Stage 2T not an acceptance criterion (Accept §4)

## Implementation Notes

- Read `GPy/kern/src/stationary_cython.pyx` / `stationary_utils.c` and the
  early Claude rewrite before inventing a third algorithm.
- For `lengthscale_grads`, the CIP description is “one matmul”; recover the
  exact layout from the prior bench or re-derive and prove parity.
- For `gradients_X`, prefer
  `grad = (tmp.sum(1)[:,None] * X - tmp @ X2) / lengthscale**2`
  (or the mathematically equivalent form that matches GPy’s `tmp` definition),
  not the memory-heavy `X[:,None,:]-X2[None,:,:]` broadcast unless harness
  peak memory is acceptable and documented.
- Keep ARD vs isotropic paths correct; isotropic already avoids Cython.

## Related

- CIP: 000E Accept decisions item 5 (Phase 1 order: stationary next)
- Files: `GPy/kern/src/stationary.py`, `stationary_cython.pyx`
- Depends on: Cholesky soak patterns / harness conventions
  (`2026-11-16_cip000E-cholesky-murray-soak`)

## Progress Updates

### 2026-11-16

Task created Ready after CIP-000E Accept (Option C). Do not start by
promoting `_lengthscale_grads_pure` / `_gradients_X_pure` to default.

### 2026-11-16 (implementation)

Landed CIP-cited matmul/GEMM forms as defaults on
`cip000E-numpy-blas-primitives`. Linux multicore confirmation still tracked
under the Cholesky soak / shared multicore note (not blocking this hotspot
given the clear Accelerate win).
