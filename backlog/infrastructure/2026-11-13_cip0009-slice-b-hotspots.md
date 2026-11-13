---
id: 2026-11-13_cip0009-slice-b-hotspots
title: CIP-0009 Slice B — next coverage hotspots after plotting
status: Ready
priority: Medium
created: '2026-11-13'
last_updated: '2026-11-13'
category: infrastructure
related_cips:
- '0009'
owner: Neil Lawrence
dependencies:
- 2026-11-13_cip0009-slice-a-plotting-coverage
tags:
- backlog
- testing
- coverage
- cip0009
---

# Task: CIP-0009 Slice B hotspot coverage

## Description

After Slice A plotting coverage lands, inventory the next 2–3 lowest-coverage
pure-Python modules under `GPy/` (Codecov project view or
`coverage report --skip-covered`) and add focused unit tests with clear APIs.

## Acceptance Criteria

- [ ] Hotspot list documented in CIP-0009 or this task (module + rough %)
- [ ] Tests added for 2–3 chosen modules
- [ ] Coverage job still informational; no merge gate change unless CIP Accept
      revises policy

## Implementation Notes

- Prefer util / models helpers over Cython-generated paths.
- Do not chase 90%+ project coverage (CIP non-goal).

## Related

- CIP: 0009
- Depends on: 2026-11-13_cip0009-slice-a-plotting-coverage

## Progress Updates

### 2026-11-13

Task created Ready, blocked on Slice A completion.
