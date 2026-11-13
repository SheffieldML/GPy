---
id: 2026-11-13_cip000D-kern-coverage-slices
title: CIP-000D — implement kern coverage slices K1–K5
status: Completed
priority: High
created: '2026-11-13'
last_updated: '2026-11-13'
category: infrastructure
related_cips:
- '000D'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- testing
- coverage
- kernels
- cip000D
---

# Task: CIP-000D kern coverage slices

## Description

Implement CIP-000D slices: deepen EQ_ODE tests, cover legacy Periodic* and
ODE_* kernels, omit GPU psi / dead `lfm1` from coverage, optional Spline.

## Acceptance Criteria

- [x] K1: EQ_ODE1/2 cross-covariance (`Kfu`/`Kuf`) + latent `Kuu` gradient checks
- [x] K2: `PeriodicExponential` / `PeriodicMatern32` / `PeriodicMatern52` in
      `check_kernel_gradient_functions`
- [x] K3: `ODE_UYC`, `ODE_st`, `ODE_t` gradient smokes (correct index layout)
- [x] K4: `.coveragerc` omits GPU psi + `lfm1.py` with CIP comment
- [x] K5: `Spline` gradient smoke
- [x] `pytest` green for touched test modules; CIP-000D status updated

## Related

- CIP: 000D
- PR: #1276 (plan); implementation on `cip000D-kern-coverage`

## Progress Updates

### 2026-11-13

CIP Accepted / In Progress; implementation started.

Slices K1–K5 landed on `cip000D-kern-coverage`. Local `GPy/kern` cover
~66% → ~80%. Marked **Completed**.
