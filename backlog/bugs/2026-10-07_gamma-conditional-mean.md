---
id: 2026-10-07_gamma-conditional-mean
title: Implement Gamma conditional_mean (and samples) for observation-space predict
status: Proposed
priority: Medium
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
- likelihood
- gamma
- predict
---

# Task: Gamma observation-space predictive moments

## Description

Follow-up to **Raashish Aggarwal**’s #1125: Gamma + Laplace training works, but
`model.predict(...)` (with likelihood) still falls through because `Gamma` does
not implement `conditional_mean` / `samples`. The October demos therefore show
latent predictions with `include_likelihood=False` only.

## Acceptance Criteria

- [ ] `Gamma.conditional_mean` (and variance if needed) implemented consistently with the mean-rate parameterization
- [ ] `predict` returns finite observation-space mean/variance on a small Gamma+Laplace example
- [ ] Unit test covering predictive moments or end-to-end `predict`

## Related

- CIP: 0006
- Contributor: Raashish Aggarwal (@raashish1601)
- PRs: #1125
