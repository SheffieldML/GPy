---
id: 2026-10-10_coregionalized-regression-docs
title: Document GPCoregionalizedRegression and MultioutputGP
status: Proposed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: documentation
related_cips: []
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
  `GPCoregionalizedRegression` documentation gaps
- [#1016](https://github.com/SheffieldML/GPy/issues/1016) — difference between
  `MultioutputGP` and `GPCoregionalizedRegression` (support question; answer
  belongs in docs, then close as support)

Open-issue triage (2026-10-10): documentation backlog.

## Acceptance Criteria

- [ ] Short comparison: when to use `GPCoregionalizedRegression` vs
      `MultioutputGP` (and ICM/LCM kernels)
- [ ] Document `set_XY` list inputs and normalizer behaviour from #1149
- [ ] Link from README or Sphinx models page
- [ ] Comment on #1099 / #1016; close when docs land

## Implementation Notes

- Prefer a concise notebook or Sphinx page over a large new guide.
- Can absorb citation-style Q&A (#856) separately; do not block on it.

## Related

- Issues: #1099, #1016
- PRs: #1149
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.
