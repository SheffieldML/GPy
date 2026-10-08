---
id: 2026-10-07_normalizer-multioutput-posterior-samples-f
title: Fix posterior_samples_f with normalizer and multiple output columns
status: Completed
priority: Low
created: '2026-10-07'
last_updated: '2026-10-08'
category: bugs
related_cips:
- '0006'
owner: Raashish Aggarwal
contributor: Raashish Aggarwal (@raashish1601)
dependencies: []
tags:
- backlog
- normalizer
- multioutput
- sampling
---

# Task: Multi-output normalizer in `posterior_samples_f`

## Description

Follow-up noted by **Raashish Aggarwal** in #1138: `posterior_samples_f` with a
normalizer and more than one output column failed in `inverse_variance` (full
covariance vs per-column std). **#1143** fixed this by using `inverse_covariance`,
matching `predict(full_cov=True)`. Cross-linked on #1138 and #1143.

## Acceptance Criteria

- [x] `posterior_samples_f` works for `GPRegression` with `Y.shape[1] > 1` and `normalizer=True`
- [x] Regression test covering multi-output + normalizer sampling
- [x] #1143 merged to `devel`

## Related

- CIP: 0006
- Contributor: Raashish Aggarwal (@raashish1601)
- PRs: #1138, #1143
