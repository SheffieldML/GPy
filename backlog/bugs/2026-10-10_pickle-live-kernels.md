---
id: 2026-10-10_pickle-live-kernels
title: Make used kernels and models pickleable for multiprocessing
status: Completed
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
- serialization
- pickle
- multiprocessing
---

# Task: Pickle live kernels (and used models)

## Description

[CIP-0006](../../cip/cip0006.md) restored *model* `save`/`load` / `to_dict` paths
(#1124, #1127, #1128, #1132). It did **not** fix pickling of kernels (or models)
after they have been attached to a GP — the failure mode reported in
[#605](https://github.com/SheffieldML/GPy/issues/605) and
[#932](https://github.com/SheffieldML/GPy/issues/932).

Users hit this with `multiprocessing`, `copy.deepcopy`, and sklearn
`GridSearchCV` / `RandomizedSearchCV` pipelines that pickle fitted kernels.
Workarounds (dill, unused-kernel copies, `to_dict`) are fragmented and still
fail in some cases.

Open-issue triage (2026-10-10): keep as backlog bug; do not close as addressed
by CIP-0006. No new CIP — root cause is a paramz pickle memento / setattr cycle;
tracked under CIP-0007 triage + this backlog task.

## Acceptance Criteria

- [x] Reproduce #605 / #932 on current `devel`
- [x] Identify the unpickleable attribute(s) (parent-link cycle / observers)
- [x] Kernels remain pickleable after use in `GPRegression` (round-trip)
- [x] Document preferred persistence: pickle vs `to_dict` / `save_model`
- [x] Comment on #605 and #932; close when fixed

## Implementation Notes

**Root cause:** pickling a used kernel serializes `_parent_` (the GP). On load,
the GP’s `__setstate__` walks `parameters` before the kernel has `_name`, and
`Parameterized.__setattr__` (when restoring `observers`) raises.

**Fix:** omit `_parent_` from the pickle memento (rebuilt by
`_connect_parameters` for whole-model pickles). GPy
`Parameterized.__getstate__` pops `_parent_` (#1165). Companion paramz 0.10.1
hardening: https://github.com/sods/paramz/pull/50.

**Persistence guidance:** use `pickle` / `copy` for live objects and
multiprocessing; prefer `to_dict` / `save_model` for durable, version-tolerant
model archives (CIP-0006 paths).

## Related

- CIP: 0006 (residual; model save/load done), 0007 (issue triage)
- Issues: #605, #932 (closed)
- PRs: GPy #1165 (merged); paramz #50 (companion)

## Progress Updates

### 2026-10-10

Task created from open-issue triage against October correctness batch.

### 2026-10-10 (execution)

Reproduced; fixed on `devel` via #1165; closed #605 and #932.
