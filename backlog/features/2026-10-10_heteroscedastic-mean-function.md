---
id: 2026-10-10_heteroscedastic-mean-function
title: Support mean_function on GPHeteroscedasticRegression
status: Completed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: features
related_cips: []
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- heteroscedastic
- mean-function
---

# Task: Heteroscedastic regression + mean function

## Description

[#875](https://github.com/SheffieldML/GPy/issues/875):
`GPHeteroscedasticRegression` does not accept `mean_function` (wrapper defaults
omit it). [#1141](https://github.com/SheffieldML/GPy/pull/1141) fixed mean
function inclusion in `predictive_gradients` for the core GP path; it does not
add the constructor kwarg on this wrapper.

Open-issue triage (2026-10-10): backlog feature.

## Acceptance Criteria

- [x] `GPHeteroscedasticRegression(..., mean_function=mf)` constructs successfully
- [x] Optimize / predict use the mean function consistently with `GPRegression`
- [x] Unit test covering construction + predict
- [x] Comment / close #875 via #1170

## Implementation Notes

- Thin wrapper change: pass `mean_function` through to `GP.__init__`.
- Pair review with heteroscedastic `set_XY` bug task (#1169).

## Related

- Issues: #875
- PRs: #1141 (related core fix)
- Related backlog: `2026-10-10_heteroscedastic-set-xy`
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

Implemented `mean_function=` on the wrapper + `test_gp_heteroscedastic_mean_function`.
