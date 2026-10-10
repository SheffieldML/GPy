---
id: 2026-10-10_studentt-observation-covariance
title: Fix negative observation covariance with Student-t likelihood
status: In Progress
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0006'
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- likelihood
- student-t
- covariance
---

# Task: Student-t observation-space covariance stays PSD

## Description

[#993](https://github.com/SheffieldML/GPy/issues/993) reports that asking for
the full *observation* output distribution (y*, not f*) under a Student-t
likelihood can yield a negative covariance (non-PSD). Reported at GPSS without
a minimal script.

Related October work: [#1145](https://github.com/SheffieldML/GPy/pull/1145)
fixed `StudentT.conditional_variance` (include `sigma2`; scalar for NumPy 2
`quad`). That is necessary but **not** sufficient for full predictive
covariance of y*.

Root cause on `devel` after #1145: `Likelihood.predictive_values` ignored
`full_cov` and passed the latent covariance matrix into 1D quadrature
(ravelled), producing NaNs / non-PSD matrices. For the identity link,
observation noise is constant in f*, so Cov(y*) = Cov(f*) + noise·I.

## Acceptance Criteria

- [x] Minimal reproduction for Student-t + full observation cov on `devel`
- [x] Confirm whether #1145 already fixes the symptom; if yes, close #993
- [x] If not: predictive observation covariance is PSD (or documented limitation)
- [x] Unit test covering the failing path
- [ ] Comment / close #993

## Implementation Notes

- Distinguish latent (f*) vs observation (y*) predictive paths in
  `Likelihood` / `GP.predict`.
- Check quadrature / moment matching when `full_cov=True`.
- #1145 did **not** fix #993; identity-link full_cov handled in `StudentT.predictive_values`.
- Base `Likelihood.predictive_values` now raises `NotImplementedError` for
  `full_cov=True` so other likelihoods fail loudly instead of silently.

## Related

- CIP: 0006, 0007
- Issues: #993
- PRs: #1145 (related); fix branch `fix/993-studentt-full-cov-observation`
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

- Reproduced: `predict(..., full_cov=True, include_likelihood=True)` with
  Student-t + Laplace yields NaN full cov after #1145.
- Implementing identity-link Cov(y*) = Cov(f*) + σ² ν/(ν−2) I.
