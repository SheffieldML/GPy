---
id: 2026-11-13_cip0009-slice-b-hotspots
title: CIP-0009 Slice B — next coverage hotspots after plotting
status: Completed
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

- [x] Hotspot list documented in CIP-0009 or this task (module + rough %)
- [x] Tests added for 2–3 chosen modules
- [x] Coverage job still informational; no merge gate change unless CIP Accept
      revises policy

## Implementation Notes

- Prefer util / models helpers over Cython-generated paths.
- Do not chase 90%+ project coverage (CIP non-goal).
- Inventory (post–Slice A suite, ~57% project): many plotting paths remain
  thin by design; util hotspots included `debug` / `subarray_and_sorting` /
  `cluster_with_offset` / `squashers` / `matplot_dep.util`.
- Root cause for several util modules: existing cases lived under `UtilTest`,
  which pytest does not collect. Rename to `TestUtil` was the highest-ROI fix.

## Related

- CIP: 0009
- Depends on: 2026-11-13_cip0009-slice-a-plotting-coverage
- Slice A PR: [#1273](https://github.com/SheffieldML/GPy/pull/1273)

## Progress Updates

### 2026-11-13

Task created Ready, blocked on Slice A completion.

Inventory + implementation on `cip0009-coverage-slice-b`: rename `UtilTest`
→ `TestUtil`, extend `checkFullRank` / `subarray` axis=0 coverage, add
`TestSquashers`. All 18 `test_util.py` tests pass. Marked **Completed**.
