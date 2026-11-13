---
id: "2026-11-13_chunked-threaded-rbf-psi"
title: "Optional chunked / threaded RBF (and linear) psi-statistics"
status: "Proposed"
priority: "Medium"
created: "2026-11-13"
last_updated: "2026-11-13"
category: "features"
related_cips: ["0010"]
owner: ""
dependencies: []
tags:
- backlog
- psi-statistics
- performance
- numpy
- deepgp
---

# Task: Chunked / threaded psi-statistics for RBF (and linear)

> **Note**: Backlog tasks are DOING the work defined in CIPs (HOW).  
> Use `related_cips` to link to CIPs. Don't link directly to requirements (bottom-up pattern).

## Description

Add **optional** row-chunking (and an optional thread pool) to GPy's RBF and
linear psi-statistic wrappers so large-N variational models can bound peak
memory and use multiple cores without MPI.

This is the GPy-side piece of PyDeepGP [CIP-0001](https://github.com/SheffieldML/PyDeepGP/blob/main/cip/cip0001.md)
(R12 / R13). It sits under GPy [CIP-0010](../../cip/cip0010.md) as the
single-machine "other" that reduces reliance on MPI (already a CIP-0010
deprecation candidate). Bayesian GPLVM benefits as well as Deep GP.

A prototype that monkey-patches `GPy.kern.src.psi_comp.rbf_psi_comp` lives in
PyDeepGP `cip/cip0001/psi_threads_prototype.py`. On a 2-core machine it matched
2 MPI ranks in speed for representative (N, M) and used far less memory by
keeping peak RSS roughly constant in N.

## Acceptance Criteria

- [ ] `psicomputations` / `psiDerivativecomputations` in
      `kern/src/psi_comp/rbf_psi_comp.py` (and `linear_psi_comp.py`) accept
      optional `chunk_size=None` and `n_threads=1` (or equivalent); defaults
      preserve today's behaviour and numerical results.
- [ ] Implementation calls the existing pure `__psi*` / `_psi*compDer`
      functions per row slice; reduces by sum (psi2, parameter grads) or
      concatenate (psi1, `dL_dmu`, `dL_dS`).
- [ ] Kernel-level `Cache_this` / in-place `.gradient` are not touched from
      multiple threads; threads run only inside the pure psi functions.
- [ ] Psi1/psi2 caching across forward and derivative for the same chunk
      avoids the prototype's ~35% single-thread overhead at small N where
      possible (R13).
- [ ] Tests: chunked vs unchunked equality for objective-relevant outputs
      and gradients (several chunk sizes, including sizes that do not divide
      N); `checkgrad` on a small Bayesian GPLVM (or equivalent) with
      chunking on.
- [ ] Docs: short note in kernel / variational docs or changelog; defaults
      remain off until many-core benchmarks justify otherwise.
- [ ] No new hard dependency (stdlib `concurrent.futures` is enough).

## Implementation Notes

- Seam: public wrappers in `rbf_psi_comp.py` already delegate to
  `__psi1computations`, `__psi2computations`, `_psi1compDer`, `_psi2compDer`.
  That is the right place for optional map-reduce; do not rewrite the
  numerics.
- Config surface: keyword args on the psi functions first; a later optional
  global/config default can follow once Deep GP / GPLVM callers need it.
- Cross-repo: PyDeepGP CIP-0001 Phase 3 (R14–R16) will consume this API for
  DeepGP inference; land the GPy PR before or with that work.
- Out of scope here: MPI removal, Theano/encoder work (PyDeepGP-local), GPU
  psi paths (separate CIP-0010 candidates).

## Related

- CIP: [0010](../../cip/cip0010.md) (GPy2 readiness — MPI deprecation / single-machine alternative)
- PyDeepGP CIP: [0001](https://github.com/SheffieldML/PyDeepGP/blob/main/cip/cip0001.md) (R12–R13)
- Prototype: https://github.com/SheffieldML/PyDeepGP/blob/main/cip/cip0001/psi_threads_prototype.py

## Progress Updates

### 2026-11-13

Task sketched from PyDeepGP CIP-0001 Accept-path planning. Status Proposed
until CIP-0010 / maintainers confirm this is the preferred GPy home for
chunking (vs only PyDeepGP-side serial chunking).
