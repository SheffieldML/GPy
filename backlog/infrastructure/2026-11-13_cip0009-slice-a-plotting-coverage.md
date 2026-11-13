---
id: 2026-11-13_cip0009-slice-a-plotting-coverage
title: CIP-0009 Slice A — plotly fill/bar/annotation and PCA / StateSpace plot smokes
status: Completed
priority: High
created: '2026-11-13'
last_updated: '2026-11-13'
category: infrastructure
related_cips:
- '0009'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- testing
- coverage
- plotting
- plotly
- cip0009
---

# Task: CIP-0009 Slice A plotting coverage

## Description

Finish CIP-0009 Slice A: extend `GPy/testing/test_plotting.py` so plotly
backend helpers and matplotlib PCA / StateSpace plot paths execute in CI.
Partial coverage already landed in [#1249](https://github.com/SheffieldML/GPy/pull/1249)
(`TestPlotly7`); this task covers the remainder called out in CIP-0009.

## Acceptance Criteria

- [x] Plotly offline tests exercise `fill_between`, `barplot`, and
      `annotation_heatmap` (skip via `pytest.importorskip` when plotly absent)
- [x] Matplotlib smoke tests call `PCA.plot_fracs` and `PCA.plot_2d`
- [x] Matplotlib smoke test calls `StateSpace.plot` (or equivalent attached
      plot helper) on a tiny fitted / constructed model
- [x] New tests pass under `poetry run pytest GPy/testing/test_plotting.py`
- [x] CIP-0009 implementation status updated for Slice A remainder

## Implementation Notes

- Follow existing `TestPlotly7` setup/teardown (`change_plotting_library`).
- Prefer direct `plotting_library()` calls for fill/bar/annotation so the
  plotly_dep branches run without needing full GP confidence plots.
- Old `GPy.models.state_space.StateSpace.plot` imports removed `models_plots`;
  use the modern `GPy.models.StateSpace` that receives `plot` from
  `GPy.plotting` attachment.
- Branch / worktree: `cip0009-coverage-slice-a` at
  `/Users/neil/SheffieldML/GPy-cip0009-coverage`.

## Related

- CIP: 0009
- PRs: #1249 (partial Slice A), #1229 (trigger)
- Documentation: `cip/cip0009.md`

## Progress Updates

### 2026-11-13

CIP-0009 Accepted and moved to In Progress. Task created; implementation
started in worktree `GPy-cip0009-coverage`.

Added `TestPlotly7` fill/bar/annotation, `TestPCAPlotSmokes`, and
`TestStateSpacePlotSmoke` (`sde_Matern32`) in `GPy/testing/test_plotting.py`.
All six new tests passed locally. Marked **Completed**.
