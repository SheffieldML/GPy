---
id: 2026-10-26_deprecate-matplotlib-visualize
title: Deprecate matplotlib visualize LVM callbacks
status: In Progress
priority: Medium
created: "2026-10-26"
last_updated: "2026-10-28"
category: infrastructure
related_cips:
- "000B"
owner: Neil Lawrence
dependencies:
- 2026-10-26_port-lvm-demos-to-explore
- 2026-10-26_plotly-skeleton-drop-vpython
tags:
- backlog
- deprecations
- plotting
---

# Task: Deprecate matplotlib visualize

> Backlog task for [CIP-000B](../../cip/cip000B.md). Last code step before docs.

## Description

Once demos no longer import `lvm` / `lvm_dimselect` / VPython, deprecate
`GPy.plotting.matplot_dep.visualize` and point callers at
[vidden](https://github.com/lawrennd/vidden). Leave geometry helpers only if
stick/skeleton views still share them; otherwise remove the file after a
warning cycle.

Do not touch static matplotlib plotting (`plot_latent`, `plot_confidence`).

## Acceptance Criteria

- [x] Importing `visualize` emits a DeprecationWarning pointing at vidden
- [x] Warning names the interactive classes, notes macOS/Jupyter matplotlib callbacks and visualisation-out-of-core, and points at [discussion #1252](https://github.com/SheffieldML/GPy/discussions/1252)
- [ ] GPy examples no longer import the old interactive classes
- [ ] VPython path gone (or already removed by the skeleton task)
- [ ] Static `GPy.plotting` matplotlib/plotly backends unchanged

## Implementation Notes

- Callers of the old API are GPy examples, not a wide public surface.
- Shrink or delete `visualize.py` only after the warning is in place and
  examples have moved.

## Related

- CIP: 000B
- Depends: 2026-10-26_port-lvm-demos-to-explore, 2026-10-26_plotly-skeleton-drop-vpython
- Code: `GPy/plotting/matplot_dep/visualize.py`

## Progress Updates

### 2026-10-26

Task created when CIP-000B was Accepted.

### 2026-10-27

Explorer moved to vidden. `visualize` import now warns and points at
https://github.com/lawrennd/vidden. Examples still call `lvm` /
`lvm_dimselect` until those demos are ported there.

### 2026-10-28

Warning text now inventories the interactive classes, states the two
reasons (macOS/Jupyter matplotlib callbacks; visualisation should not
live in the core library), and invites comment on
[discussion #1252](https://github.com/SheffieldML/GPy/discussions/1252).
Constructors of `data_show` subclasses and `data_play` warn as well.
