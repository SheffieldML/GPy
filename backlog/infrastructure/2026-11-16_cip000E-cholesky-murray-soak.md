---
id: 2026-11-16_cip000E-cholesky-murray-soak
title: CIP-000E — Cholesky Murray blocked soak and multicore confirmation
status: In Progress
priority: High
created: "2026-11-16"
last_updated: "2026-11-16"
category: infrastructure
related_cips:
- "000E"
- "000F"
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- cip000E
- cython
- cholesky
- murray
- benchmarks
---

# Task: CIP-000E Cholesky Murray blocked soak

## Description

Finish the Cholesky backprop replacement under CIP-000E **Option C**.

The wired default is already Murray Level-3 blocked reverse
(`GPy.util.choleskies.cholesky_vjp_blocked` → `backprop_gradient`),
from Murray (2016) arXiv:1602.07527. That is the algorithm cited in the
CIP benchmark table (**0.15× / 0.07×** vs Cython on the original Linux run;
**0.46× / 0.26× / 0.23×** at N=500/1000/2000 on macOS/Accelerate).

### Process constraint (do not regress)

A column-matvec NumPy form (`cholesky_vjp`) was briefly treated as “the”
BLAS replacement. It matches the oracle numerically but **loses to Cython**
on macOS/Accelerate (~2–4× slower). That form is a useful reference /
simpler primitive only. **It must not become the default again.**

Any future change to the default must cite harness output where the
candidate’s ×cython is **&lt; 1** on at least one multicore host with a
named BLAS (Accelerate, OpenBLAS, or MKL), at N ∈ {1000, 2000}.

## Acceptance Criteria

- [x] `backprop_gradient is cholesky_vjp_blocked` when the package loads
- [x] Parity tests: Murray blocked vs pure oracle vs Cython
      (`GPy/testing/test_cython.py::TestCholeskiesBackprop`)
- [x] In-repo harness includes `murray` vs `cython` vs `vjp`
      (`benchmarks/cip000E/run_cholesky_backprop.py`)
- [ ] Re-run harness on **Linux multicore** with OpenBLAS or MKL; record
      wall times, ×cython, peak tracemalloc, BLAS name, and
      `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS`
      variants (1 and default) in CIP-000E or `benchmarks/cip000E/` notes
- [x] SVGP tests green with Murray default (`pytest GPy/testing/test_svgp.py`)
- [x] Escape hatch documented: `choleskies_cython.backprop_gradient_par_c`
      remains importable; not the wired default
- [x] macOS threading spot-check (default threads vs
      `OMP/OPENBLAS/VECLIB/MKL_NUM_THREADS=1` at N=1000): Murray stays ahead
      (~0.33× / ~0.28×). Linux multicore row still required before soak-complete.
- [ ] Explicit note in CIP-000E Implementation Status that Cholesky Phase 1
      item is soak-complete only after the Linux multicore row exists

## Implementation Notes

- Algorithm SoT: Murray blocked reverse (`chol_blocked_rev`), not Smith
  unblocked loops and not ad-hoc column `dsymv` wrappers that fail the
  ×cython &lt; 1 bar.
- Keep `cholesky_vjp` (column form) as a named primitive for comparison and
  CIP-000F Stage 2 experiments; do not delete it during soak.
- Block size `N_b=128` is the current default; re-tune on Linux if
  harness shows a better NB, and record the choice.
- Flat/triang Cython maps are **out of scope** here (separate backlog).

## Related

- CIP: 000E (Accept Option C, 2026-11-16)
- CIP: 000F Stage 2 (pure numerical primitives)
- Branch: `cip000E-numpy-blas-primitives`
- Discussion: [#1279](https://github.com/SheffieldML/GPy/discussions/1279)
- Paper: https://arxiv.org/abs/1602.07527

## Progress Updates

### 2026-11-16

Murray blocked landed and wired as default on
`cip000E-numpy-blas-primitives`. macOS/Accelerate harness confirms win vs
Cython. Linux multicore + threading matrix still outstanding before calling
this hotspot soak-complete.

### 2026-11-16 (continued)

SVGP tests passed. macOS thread-env spot-check recorded. NB=128/256 both win
at N=2000 when the machine is quiet (noisy outlier earlier); default NB remains
128. Linux OpenBLAS/MKL matrix still open.
