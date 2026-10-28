---
id: 2026-10-26_plotly-skeleton-drop-vpython
title: Plotly skeleton view and drop VPython
status: Ready
priority: Medium
created: "2026-10-26"
last_updated: "2026-10-26"
category: features
related_cips:
- "000B"
owner: Neil Lawrence
dependencies:
- 2026-10-26_explore-module-optional-extra
tags:
- backlog
- plotting
- plotly
- mocap
---

# Task: Plotly skeleton view; drop VPython

> Backlog task for [CIP-000B](../../cip/cip000B.md).

## Description

Replace `skeleton_show` / `mocap_data_show_vpython` with Plotly 3D line
segments using existing `connect` / `skel` data. Plotly is already
`gpy[plotting]`. Do not add pythreejs, k3d, or PyVista unless Plotly cannot
draw the bones.

Remove the `import visual` path from GPy.

## Acceptance Criteria

- [ ] `skeleton_view` draws mocap bones from a reconstructed `Y` row
- [ ] `cmu_mocap` demo uses the new view (or skips if Plotly is missing)
- [ ] No remaining `import visual` / `visual_available` in GPy
- [ ] No new 3D dependency beyond Plotly

## Implementation Notes

- Start from `visualize.skeleton_show` / `mocap_data_show` geometry, not from
  VPython scene code.
- 2D `stick_show` connectivity stays in the explore stick view.

## Related

- CIP: 000B
- Depends: 2026-10-26_explore-module-optional-extra
- Demo: `GPy.examples.dimensionality_reduction.cmu_mocap`

## Progress Updates

### 2026-10-26

Task created when CIP-000B was Accepted.
