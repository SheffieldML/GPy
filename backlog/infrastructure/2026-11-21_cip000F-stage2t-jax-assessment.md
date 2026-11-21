---
id: 2026-11-21_cip000F-stage2t-jax-assessment
title: CIP-000F Stage 2T — experimental JAX LML assessment (out of package)
status: Ready
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

- [ ] At least one experimental JAX LML (exact GP / RBF) with numerical and
      gradient parity notes
- [ ] Barriers classified and checked into CIP notes or a linked report
- [ ] No JAX dependency added to GPy package metadata
- [ ] Findings linked for CIP-0010 transformation-readiness report

## Related

- CIP: 000F, 0010

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.
