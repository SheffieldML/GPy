---
id: 2026-10-10_pickle-live-kernels
title: Make used kernels and models pickleable for multiprocessing
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
by CIP-0006.

## Acceptance Criteria

- [ ] Reproduce #605 / #932 on current `devel` with a minimal script
- [ ] Identify the unpickleable attribute(s) (observers, caches, Cython state)
- [ ] Kernels remain pickleable after use in `GPRegression` (round-trip)
- [ ] Document preferred persistence: pickle vs `to_dict` / `save_model`
- [ ] Comment on #605 and #932; close when fixed or with a clear wontfix + docs

## Implementation Notes

- Prefer fixing `__getstate__` / `__setstate__` (or dropping non-essential
  runtime links) over forcing every caller onto `to_dict`.
- Guard against regressing CIP-0006 serialization tests.
- Multiprocessing is the primary motivator; sklearn pipelines are a secondary
  check.

## Related

- CIP: 0006 (residual; model save/load done)
- Issues: #605, #932 (also #535 historically)
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage against October correctness batch.
