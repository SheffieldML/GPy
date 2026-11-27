---
id: 2026-11-27_sparse-kuu-jitter
title: Make the Kuu jitter of VarDTC and FITC a constructor option
status: Completed
priority: Medium
created: "2026-11-27"
last_updated: "2026-11-27"
category: features
owner: Neil Lawrence
dependencies:
- 2026-11-27_exact-gaussian-jitter
tags:
- backlog
- inference
- numerics
---

# Task: Configurable Kuu jitter in the sparse inference methods

## Description

VarDTC adds 1e-8 to the diagonal of Kuu, FITC adds 1e-6, and GPflow, GPJax and GPmat use 1e-6.
In gp-scuderia the difference shows as about 2e-5 relative in DTCVar gradients and about 1e-2 in
values at M=1000 when Kuu is ill conditioned. It was traced to the jitter against an independent
numpy implementation.

`const_jitter` was a class attribute, so it could be set on an instance but not passed to the
constructor or serialized. Make it a constructor argument of `VarDTC` and `FITC`, with the defaults
unchanged, and record it in `VarDTC.to_dict`.

## Acceptance Criteria

- [x] `VarDTC(const_jitter=...)` and `FITC(const_jitter=...)`; defaults 1e-8 and 1e-6 unchanged.
- [x] `VarDTC.to_dict` records the jitter and `from_dict` restores it.
- [x] Docstrings state the convention differences.
- [x] Tests show the option is used and the defaults are unchanged.

## Implementation Notes

`DTC` has a `const_jitter` attribute that its inference code never reads, so it is left alone.
VarDTC also uses `const_jitter` as the floor on the noise variance in the noise precision; this is
noted in its docstring. `FITC` has no `to_dict`.

## Related

- gp-scuderia: after a GPy release, the adapter can pass `const_jitter=1e-6` to VarDTC and drop the
  `dtcvar_*` known cases.

## Progress Updates

### 2026-11-27
Created and implemented with tests in `GPy/testing/test_sparse_kuu_jitter.py`.
