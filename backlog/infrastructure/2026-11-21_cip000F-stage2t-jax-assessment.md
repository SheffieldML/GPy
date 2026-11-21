---
id: 2026-11-21_cip000F-stage2t-jax-assessment
title: CIP-000F Stage 2T — experimental JAX LML assessment (out of package)
status: Completed
priority: Medium
created: "2026-11-21"
last_updated: "2026-11-21"
category: infrastructure
related_cips:
- "000F"
- "0010"
owner: Neil Lawrence
dependencies:
- 2026-11-21_cip000F-stage2-pure-numerical
tags:
- backlog
- cip000F
- jax
- transformation-readiness
---

# Task: CIP-000F Stage 2T JAX assessment

## Description

After the first pure LML (or similar) extraction, implement an **out-of-package**
experimental JAX equivalent. Record barrier classifications (formulation /
precision / control-flow / …) for CIP-0010 Workstream G. Do **not** add JAX
as a GPy 1 runtime dependency. Successful `jax.jit` / `jax.grad` is evidence,
not a Stage 2 gate.

## Acceptance Criteria

- [x] At least one experimental JAX LML (exact GP / RBF) with numerical and
      gradient parity notes
- [x] Barriers classified and checked into CIP notes or a linked report
- [x] No JAX dependency added to GPy package metadata
- [x] Findings linked for CIP-0010 transformation-readiness report

## Related

- CIP: 000F, 0010
- Notes: `cip/cip000F/notes/stage2t-jax-lml.md`
- Experiment: `cip/cip000F/experiments/exact_gp_rbf_lml_jax.py`

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.

### 2026-11-21 (later)

Completed: JAX RBF exact-LML with float64 parity (~1e-15), `jax.grad` vs FD
(~1e-7), `jax.jit` on scalar LML. Barriers recorded. No package dependency.
