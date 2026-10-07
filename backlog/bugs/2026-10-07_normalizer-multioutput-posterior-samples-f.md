---
id: 2026-10-07_normalizer-multioutput-posterior-samples-f
title: Fix posterior_samples_f with normalizer and multiple output columns
status: Proposed
priority: Low
created: '2026-10-07'
last_updated: '2026-10-07'
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
normalizer and more than one output column already fails in `inverse_variance`
(full covariance does not broadcast with the per-column std). That path was left
out of #1138; single-output normalizer samples and LPD are fixed.

## Acceptance Criteria

- [ ] `posterior_samples_f` works for `GPRegression` with `Y.shape[1] > 1` and `normalizer=True`
- [ ] Regression test covering multi-output + normalizer sampling
- [ ] Documented interaction with `posterior_samples` if behaviour differs

## Related

- CIP: 0006
- Contributor: Raashish Aggarwal (@raashish1601)
- PRs: #1138
