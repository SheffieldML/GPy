---
id: 2026-10-10_plotting-return-api
title: Clarify plotting return API (plot / add_to_canvas / show)
status: Done
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: features
related_cips:
- '0004'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- plotting
- matplotlib
- api
---

# Task: Plotting return API (`plot` / `add_to_canvas` / `show`)

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster E, [#989](https://github.com/SheffieldML/GPy/pull/989)
proposed that `add_to_canvas` return `ax`. Separately,
[#920](https://github.com/SheffieldML/GPy/issues/920) shows tutorials and
`GPy.plotting.show` break because `GPRegression.plot()` returns a **dict** of
plots, not a matplotlib `Figure`.

[#1153](https://github.com/SheffieldML/GPy/pull/1153) fixed matplotlib ≥ 3.4
`_process_unit_info` only — it does not change return types.

## Design note (chosen: Option B)

Keep `add_to_canvas` / `model.plot()` returning the **plots dict** (callers may
need artists). Teach matplotlib `show_canvas` / `GPy.plotting.show` to accept:

1. matplotlib `Axes`
2. matplotlib `Figure`
3. the plots `dict` (resolve `.axes.figure` from nested artists)

Do **not** change the return type of `add_to_canvas` to `ax` (#989 approach), to
avoid breaking code that uses the plots dict.

## Acceptance Criteria

- [x] Short design note: return type of `plot` / `add_to_canvas` stays dict;
      `show` resolves Figure from dict/Axes/Figure
- [x] `show` accepts dict return (tutorials / #920)
- [x] Tests for the chosen API (`TestShowAcceptsPlotDict`)
- [x] Disposition: close #989 as superseded; close #920 when fix lands
- [ ] Optional: Plotly deprecation path for [#968](https://github.com/SheffieldML/GPy/issues/968)
      as a follow-up checklist item

## Implementation Notes

- Prefer minimal breakage: helper to extract axes/figure from the dict may be
  enough for `show` and notebooks.
- Avoid coupling to CIP-0003 packaging; plotting API is behavioural.

## Related

- CIP: 0004
- Issues: #920, #968 (follow-up)
- PRs: #989 (source idea, superseded); #1153 (orthogonal matplotlib fix)
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage / CIP-0004 leftover.

Option B implemented: `MatplotlibPlots._resolve_figure` +
`TestShowAcceptsPlotDict`; close #989/#920 with the landing PR.
