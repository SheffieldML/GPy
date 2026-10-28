---
id: 2026-10-26_port-lvm-demos-to-explore
title: Port stick, oil, and faces LVM demos to explore
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
- examples
- gplvm
---

# Task: Port LVM demos to explore

> Backlog task for [CIP-000B](../../cip/cip000B.md).

## Description

Switch dimensionality-reduction demos off matplotlib `visualize.lvm` /
`lvm_dimselect` and onto `GPy.plotting.explore`. Cover:

- `stick` / `stick_bgplvm` (stick view)
- `bgplvm_oil` / oil dimselect (vector view)
- Olivetti faces (image view)
- `data_play` / `stick_play` as a slider or periodic Panel callback, not
  `time.sleep`

Leave CMU mocap / 3D skeleton until the Plotly skeleton task.

## Acceptance Criteria

- [ ] Listed demos reconstruct from a latent pointer without matplotlib GUI
      backends
- [ ] `plot=True` still no-ops or skips cleanly when the explore extra is missing
- [ ] `stick_play` no longer depends on `data_play` + `time.sleep`
- [ ] Demos do not import `visualize.lvm` or `lvm_dimselect`

## Implementation Notes

- `dimensionality_reduction.py` currently calls `prepare_gui` / `run_gui` only
  on the doodle branch; on `main` it still uses raw `visualize` callbacks.
- Image layout for Olivetti is `(112, 92)`; oil is a 1D vector.

## Related

- CIP: 000B
- Depends: 2026-10-26_explore-module-optional-extra
- Code: `GPy/examples/dimensionality_reduction.py`

## Progress Updates

### 2026-10-26

Task created when CIP-000B was Accepted.
