---
id: 2026-10-10_coregionalized-regression-docs
title: Document GPCoregionalizedRegression and MultioutputGP
status: In Progress
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: documentation
related_cips:
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- documentation
- coregionalization
- multioutput
---

# Task: Coregionalized / multi-output model docs

## Description

After [#1149](https://github.com/SheffieldML/GPy/pull/1149) (normalizer +
list-aware `set_XY` on coregionalized models), users still lack clear docs:

- [#1099](https://github.com/SheffieldML/GPy/issues/1099) —
  `GPCoregionalizedRegression` documentation gaps (especially predict +
  `Y_metadata` / `output_index`)
- [#1016](https://github.com/SheffieldML/GPy/issues/1016) — difference between
  `MultioutputGP` and `GPCoregionalizedRegression` (closed with a short
  comparison; fuller text belongs in docs)

## Acceptance Criteria

- [x] Short comparison: when to use `GPCoregionalizedRegression` vs
      `MultioutputGP` (and ICM/LCM kernels)
- [x] Document `set_XY` list inputs and normalizer behaviour from #1149
- [x] Link from Sphinx (`doc/source/tuto_coregionalized.rst` + index)
- [ ] Comment / close #1099 when docs land

## Implementation Notes

- Added list-aware `predict` / `predict_noiseless` / `predict_quantiles` on
  coregionalized models via `util.multioutput.prepare_Xnew` (parity with
  `MultioutputGP`) so the documented one-output predict pattern works.

## Related

- Issues: #1099, #1016
- PRs: #1149
- Branch: `docs/1099-coregionalized-docs`

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

### 2026-10-10 (execution)

Sphinx page + list `predict` + regression test on `docs/1099-coregionalized-docs`.
