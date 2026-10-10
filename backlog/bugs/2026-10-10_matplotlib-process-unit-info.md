---
id: 2026-10-10_matplotlib-process-unit-info
title: Fix _process_unit_info for matplotlib ≥ 3.4
status: In Progress
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0004'
owner: Neil Lawrence
contributor: timovwb (@timovwb)
dependencies: []
tags:
- backlog
- plotting
- matplotlib
---

# Task: matplotlib ≥ 3.4 `_process_unit_info`

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster E (plotting), land the fix from
[#960](https://github.com/SheffieldML/GPy/pull/960) (Timo Vanwynsberghe / @timovwb)
for [#953](https://github.com/SheffieldML/GPy/issues/953).

matplotlib 3.4 changed `Axes._process_unit_info` so keyword args `xdata` / `ydata`
raise `TypeError`. Current `devel` still uses the old call style in:

- `GPy/plotting/matplot_dep/plot_definitions.py`
- `GPy/plotting/matplot_dep/base_plots.py` (same bug; not in #960)

Do **not** merge #960 as-is: also fix `base_plots.py`, and raise the plotting
extra to `matplotlib >= 3.4` (as requested in review on #960). With
`python_requires >= 3.9` that floor is appropriate.

**Out of scope here:** [#989](https://github.com/SheffieldML/GPy/pull/989)
(`add_to_canvas` return value) — related CIP-0004 plotting cluster item, needs its
own design note / backlog task.

## Acceptance Criteria

- [x] Both call sites use the matplotlib ≥ 3.4 `_process_unit_info` API
- [x] Plotting extra requires `matplotlib >= 3.4`
- [ ] Credits #960 / @timovwb; close #960 (and #953 if fixed) when the PR merges

## Implementation Notes

New call shape (matplotlib 3.4+):

```python
ax._process_unit_info([("x", X), ("y", y1)], convert=False)
ax._process_unit_info([("y", y2)], convert=False)
```

## Related

- CIP: 0004
- Contributor: timovwb (@timovwb)
- PRs: #960 (source idea)
- Issues: #953

## Progress Updates

### 2026-10-10

- Confirmed both call sites still broken on `devel`.
- Backlog created; maintainer PR updates both sites and bumps plotting extra to ≥3.4.
- Local smoke + `test_plotting` passed.
