---
id: 2026-10-10_plotting-return-api
title: Clarify plotting return API (plot / add_to_canvas / show)
status: Proposed
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

Open-issue triage (2026-10-10): design + tests; do not merge #989 without an
API note.

## Acceptance Criteria

- [ ] Short design note in this task (or CIP update): return type of `plot` /
      `add_to_canvas` / `show` for matplotlib backend
- [ ] Either restore Figure-compatible behaviour or document dict return and fix
      tutorials / `show`
- [ ] Tests for the chosen API
- [ ] Disposition on #989 (rebase vs re-implement vs close) and #920
- [ ] Optional: Plotly deprecation path for [#968](https://github.com/SheffieldML/GPy/issues/968)
      as a follow-up checklist item

## Implementation Notes

- Prefer minimal breakage: helper to extract axes/figure from the dict may be
  enough for `show` and notebooks.
- Avoid coupling to CIP-0003 packaging; plotting API is behavioural.

## Related

- CIP: 0004
- Issues: #920, #968 (follow-up)
- PRs: #989 (source idea); #1153 (orthogonal matplotlib fix)
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage / CIP-0004 leftover.
