---
id: 2026-10-10_studentt-observation-covariance
title: Fix negative observation covariance with Student-t likelihood
status: Proposed
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0006'
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
`quad`). That is necessary but may not be sufficient for full predictive
covariance of y*.

Open-issue triage (2026-10-10): backlog bug; verify whether #1145 already
clears #993 before investing further.

## Acceptance Criteria

- [ ] Minimal reproduction for Student-t + full observation cov on `devel`
- [ ] Confirm whether #1145 already fixes the symptom; if yes, close #993
- [ ] If not: predictive observation covariance is PSD (or documented limitation)
- [ ] Unit test covering the failing path
- [ ] Comment / close #993

## Implementation Notes

- Distinguish latent (f*) vs observation (y*) predictive paths in
  `Likelihood` / `GP.predict`.
- Check quadrature / moment matching when `full_cov=True`.

## Related

- CIP: 0006
- Issues: #993
- PRs: #1145 (related)
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.
