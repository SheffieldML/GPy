---
id: 2026-11-21_cip000F-stage2n-numerical-interfaces
title: CIP-000F Stage 2N — conceptual CovarianceOperator / ConditionedState sketch
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
- architecture
- transformation-readiness
---

# Task: CIP-000F Stage 2N numerical interfaces

## Description

With the first exact-GP extraction, sketch conceptual
`CovarianceOperator` / `ConditionedState` / `ParameterSpec` boundaries and
demonstrate condition-once / predict-many via pure state. Per Accept: dense
reference only — **no** in-tree matvec operator hierarchy in GPy 1.

## Acceptance Criteria

- [x] Conceptual interfaces documented in CIP-000F (or `cip/cip000F/` notes)
- [x] Exact-GP condition-once / predict-many demonstrated with pure state
- [x] No large operator hierarchy merged into GPy 1
- [x] Findings feed CIP-0010 Workstream G / H

## Related

- CIP: 000F, 0010
- Notes: `cip/cip000F/notes/stage2n-numerical-interfaces.md`
- Demo: `cip/cip000F/experiments/exact_gp_conditioned_numpy.py`

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.

### 2026-11-21 (later)

Completed: protocol sketch + NumPy condition-once/predict-many parity vs
`GPRegression.predict(include_likelihood=False)`. Stage 3 should prefer
posterior-scoped state; version-counter interim OK.
