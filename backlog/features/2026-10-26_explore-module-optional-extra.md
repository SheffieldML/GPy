---
id: 2026-10-26_explore-module-optional-extra
title: Add GPy.plotting.explore module and optional extra
status: Abandoned
priority: High
created: "2026-10-26"
last_updated: "2026-10-27"
category: features
related_cips:
- "000B"
owner: Neil Lawrence
dependencies:
- 2026-10-26_panel-stick-explorer-spike
tags:
- backlog
- plotting
- packaging
- holoviews
---

# Task: Explore module and optional extra

> Backlog task for [CIP-000B](../../cip/cip000B.md). Depends on the Panel spike.

## Description

Extract the spike into a small public module so demos share one explorer.
Keep GPy models ignorant of the UI: latent point in, `predict` out, viewer
pane for `Y`.

Provisional names (lock them here if the spike suggests better ones):

- `GPy.plotting.explore.latent_explorer`
- `image_view`, `vector_view`, `stick_view`

Add a Poetry extra (e.g. `gpy[explore]`) for HoloViews and Panel. Default
`pip install gpy` must not pull them.

## Acceptance Criteria

- [ ] Module importable when the extra is installed
- [ ] Import of `GPy` and `GPy.plotting` succeeds without HoloViews/Panel
- [ ] Unit tests for reconstruct helpers on `Agg` with a tiny GPLVM
- [ ] Explorer tests skip via `pytest.importorskip` when the extra is absent
- [ ] Extra declared in `pyproject.toml`

## Implementation Notes

- Do not add HoloViews/Panel to core or to `gpy[plotting]` (matplotlib/plotly).
- Tests must not start a GUI or a browser driver.
- Stick view can live here; Plotly 3D skeleton is a later task.

## Related

- CIP: 000B
- Depends: 2026-10-26_panel-stick-explorer-spike

## Progress Updates

### 2026-10-26

Task created when CIP-000B was Accepted.

### 2026-10-27

Abandoned. The explorer is not a GPy extra. It lives in
https://github.com/lawrennd/vidden (vidden CIP-0001).
