---
id: 2026-10-10_coregionalized-regression-docs
title: Document GPCoregionalizedRegression and MultioutputGP
status: Completed
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

Document `GPCoregionalizedRegression` vs `MultioutputGP`, ICM/LCM, predict
`Y_metadata`, and list-aware `set_XY` / normalizer (#1149).

## Acceptance Criteria

- [x] Short comparison: `GPCoregionalizedRegression` vs `MultioutputGP`
- [x] Document `set_XY` list inputs and normalizer behaviour from #1149
- [x] Link from Sphinx (`doc/source/tuto_coregionalized.rst` + index)
- [x] Comment / close #1099

## Implementation Notes

Landed in #1172: Sphinx page plus list-aware `predict` on coregionalized models.

## Related

- Issues: #1099, #1016 (closed)
- PR: #1172 (merged)

## Progress Updates

### 2026-10-10

Task created; completed via #1172.
