---
id: 2026-10-10_periodic-kernel-serialization
title: Serialize Periodic kernels and Coregionalize rank
status: Completed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: features
related_cips:
- '0004'
owner: Neil Lawrence
contributor: gehbiszumeis (@gehbiszumeis)
dependencies: []
tags:
- backlog
- serialization
- kernels
- periodic
- coregionalize
---

# Task: Periodic kernel `to_dict` / `from_dict` (+ Coregionalize `rank`)

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster B (serialization), land the useful
parts of [#976](https://github.com/SheffieldML/GPy/pull/976) (@gehbiszumeis):

1. `to_dict` / `from_dict` for the `Periodic*` Fourier-subspace kernels
2. Persist `Coregionalize.rank` so round-trips with `rank != 1` succeed when `W`
   is restored

Do **not** merge #976 as-is: it only added `to_dict` for `PeriodicMatern32`,
omitted `n_freq` / `lower` / `upper` from the saved dict, and mixed in unrelated
docstring typo edits. Clean re-implement covers all three Periodic subclasses
and a round-trip test (including Coregionalize `rank=2`).

## Acceptance Criteria

- [x] `PeriodicExponential`, `PeriodicMatern32`, `PeriodicMatern52` serialize
- [x] Saved dict includes `n_freq`, `lower`, `upper`
- [x] `Coregionalize.to_dict` includes `rank`; `rank>1` round-trips
- [x] Credits #976 / @gehbiszumeis; close #976 as superseded when the PR merges

## Related

- CIP: 0004
- Contributor: gehbiszumeis (@gehbiszumeis)
- PRs: #976 (source idea), #1156 (landed)

## Progress Updates

### 2026-10-10

- Assessed #976; backlog created; implementing maintainer PR.
- Landed via [#1156](https://github.com/SheffieldML/GPy/pull/1156); closed #976 as superseded.
