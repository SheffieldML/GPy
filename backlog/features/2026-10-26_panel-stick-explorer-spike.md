---
id: 2026-10-26_panel-stick-explorer-spike
title: Panel spike for stick_bgplvm latent explorer
status: Ready
priority: High
created: "2026-10-26"
last_updated: "2026-10-26"
category: features
related_cips:
- "000B"
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- plotting
- panel
- holoviews
- gplvm
---

# Task: Panel spike for stick_bgplvm

> Backlog task for [CIP-000B](../../cip/cip000B.md). No public API yet.

## Description

Prove that a browser explorer can replace matplotlib `lvm_dimselect` on macOS.
Build a Panel app around `stick_bgplvm(optimize=False)`: latent scatter, stick
reconstruction from `model.predict`, and two widgets to choose which latent
axes to plot. Other latent dimensions stay at the posterior mean.

Do this in an example or notebook on this branch. Do not add
`GPy.plotting.explore` or a Poetry extra until the spike works.

## Acceptance Criteria

- [ ] Pointer (or tap) in the latent scatter updates the stick figure via `predict`
- [ ] Two dim selectors change which latent axes are plotted
- [ ] Works in Jupyter on macOS without the matplotlib `macosx` GUI backend
- [ ] Same app runs with `panel serve`
- [ ] No new core GPy dependency

## Implementation Notes

- HoloViews `PointerXY` / `Tap` plus Panel layout, as sketched in CIP-000B.
- BayesianGPLVM latents are `m.X.mean`; copy to a writable `(1, Q)` array
  before writing the two active coordinates.
- Reuse stick connectivity from `pods.datasets.osu_run1()["connect"]`.
- Keep the spike throwaway-friendly; the next task extracts the module.

## Related

- CIP: 000B
- Demo: `GPy.examples.dimensionality_reduction.stick_bgplvm`

## Progress Updates

### 2026-10-26

Task created when CIP-000B was Accepted.
