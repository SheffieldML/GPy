---
id: 2026-10-09_fitc-multioutput-dimensions
title: Fix FITC inference for multiple output dimensions
status: Completed
priority: Medium
created: '2026-10-09'
last_updated: '2026-10-09'
category: bugs
related_cips:
- '0004'
owner: Neil Lawrence
contributor: monabf (@monabf)
dependencies: []
tags:
- backlog
- fitc
- sparse-gp
- multioutput
- gradients
---

# Task: FITC multi-output gradient fix

## Description

Revive [#821](https://github.com/SheffieldML/GPy/pull/821) (monabf, 2020) under
[CIP-0004](../../cip/cip0004.md) cluster C. FITC sparse GP regression was working
for multiple input dimensions but only a single output dimension. Gradients force
`v` through `reshape(-1, 1)`, which flattens multi-output `v` of shape `(M, D)`
and breaks `tdot(v)` / `v @ Y.T`.

Kept authorship with **@monabf**: rebased their PR branch onto `devel` rather than
re-implementing on a maintainer branch.

Plan / triage comment:
https://github.com/SheffieldML/GPy/pull/821#issuecomment-6086024438

## Acceptance Criteria

- [x] Rebase `fix_fitc_multiout` onto current `devel`
- [x] Resolve `GPy/testing/fitc.py` conflict (pytest style + multi-output test)
- [x] Inference fix (`tdot(v)` / `np.dot(v, Y.T)`) present; PR mergeable
- [x] CI green on #821
- [x] #821 merged to `devel` (2026-10-09)

## Implementation Notes

Only conflict was the test file (unittest → pytest on `devel`). Resolution kept
`devel` style and monabf’s `Y2D2D` / `test_fitc_2d2d`. Inference hunk applied
cleanly. Force-pushed rebased commit to `monabf:fix_fitc_multiout`; author
remains `and <buisson-fenet@is.mpg.de>`.

## Related

- CIP: 0004
- Contributor: monabf (@monabf)
- PRs: #821 (merged)

## Progress Updates

### 2026-10-09

- Assessed #821 as implementable; backlog created; plan commented on PR.
- Rebased contributor branch onto `devel`, resolved test conflict, force-pushed
  to fork. PR became MERGEABLE; CI green across platforms.
- #821 merged to `devel`; backlog marked Completed.
