---
id: 2026-10-10_optimize-restarts-ep-state
title: Diagnose optimize_restarts worse than optimize under EP
status: Completed
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0007'
owner: Neil Lawrence
contributor: olamarre (@olamarre)
dependencies: []
tags:
- backlog
- optimize
- expectation-propagation
- classification
---

# Task: `optimize_restarts` vs `optimize` under EP

## Description

[#1109](https://github.com/SheffieldML/GPy/issues/1109) (@olamarre):
`optimize_restarts` should be a superset of `optimize` (same start plus
random restarts) but can yield a *worse* fit. Martin Bubel traced that kernel
hyperparameters match while **ExpectationPropagation** variational /
site parameters differ between the two runs.

## Acceptance Criteria

- [x] Minimal reproduction from #1109 on current `devel`
- [x] Root cause documented (EP re-init, random state, site parameter copy)
- [x] Fix so `optimize_restarts` refreshes inference after restoring best hypers
- [x] Regression test
- [x] Comment / close #1109

## Implementation Notes

Root cause: `paramz.Model.optimize_restarts` restores only `optimizer_array`
from the best run. EP in `alternated` mode caches `_ep_approximation` for the
duration of each `optimize()` call. After several restarts, hypers come from
the best run while sites remain from the **last** restart — mismatched state,
poor predictions / log-likelihood.

Fix: `GP.optimize_restarts` calls `inference_method.on_optimization_start()`
and `parameters_changed()` after the paramz restore so EP is recomputed at the
selected hyperparameters.

## Related

- Issues: #1109
- Contributors: olamarre (@olamarre); investigation notes by Martin Bubel
- CIP: 0007
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

- Reproduced: same RBF hypers as single `optimize`, different EP `tau` sum,
  worse `log_likelihood` and predictions until EP refresh.
- Implemented `GP.optimize_restarts` refresh; regression in `test_model.py`.
