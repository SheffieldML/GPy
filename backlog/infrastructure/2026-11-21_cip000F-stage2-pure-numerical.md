---
id: 2026-11-21_cip000F-stage2-pure-numerical
title: CIP-000F Stage 2 — pure numerical functions (kernels, likelihoods)
status: Completed
priority: High
created: "2026-11-21"
last_updated: "2026-11-21"
category: infrastructure
related_cips:
- "000F"
- "000E"
- "0010"
owner: Neil Lawrence
dependencies:
- 2026-11-21_cip000F-stage0-import-boundary
tags:
- backlog
- cip000F
- paramz
- transformation-readiness
---

# Task: CIP-000F Stage 2 pure numerical functions

## Description

Extract `K` / `Kdiag`, psi-statistics, `logpdf`, and moments to module-level
functions of plain ndarrays. Prefer transformation-ready signatures. Reuse
CIP-000E primitives (`stationary_ops`, Cholesky Murray, `coregionalize_ops`)
rather than re-extracting those hotspots. Gradients stay for now. Split into
per-family PRs as needed (stationary first).

## Acceptance Criteria

- [x] Representative kernels/likelihoods delegate to pure ndarray functions
- [x] Golden-value tests for K, Kdiag, logpdf, LML and gradients recorded
- [x] Extracted functions callable without a GPy model or paramz `Param`
- [x] CIP-000E primitives reused where they already cover the hotspot
- [x] Public model API unchanged

## Related

- CIP: 000F, 000E, 0010

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.

### 2026-11-21 (later)

Started stationary / RBF slice: `rbf_ops` (`k_of_r`, `dk_dr`, `K`,
`kdiag`) and `stationary_ops` distance helpers; RBF / Stationary wrappers
delegate; golden tests in `test_cip000F_stage2_rbf_ops.py`. Next: more
stationary kernels, then a pure LML objective, then likelihoods.

### 2026-11-21 (Stage 2 acceptance)

Completed representative Stage 2 slice:

- `stationary_cov_ops` for Exponential / OU / Matern32 / Matern52
- `gaussian_ops` for Gaussian `logpdf_link` / `dlogpdf_dlink`
- `exact_gp_ops.log_marginal_and_gradients` + `rbf_log_marginal`; wired through
  `ExactGaussianInference`
- Golden parity in `test_cip000F_stage2_pure_numerical.py`

CIP-000F Stage 2 production acceptance criteria are met. Optional follow-ups
(remaining kernels, psi-statistics) are not gates on Stage 2 close-out.
