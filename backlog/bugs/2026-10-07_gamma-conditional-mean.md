---
id: 2026-10-07_gamma-conditional-mean
title: Implement Gamma conditional_mean (and samples) for observation-space predict
status: Completed
priority: Medium
created: '2026-10-07'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0006'
owner: Neil Lawrence
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

Follow-up chain:

1. **#1125** (Raashish) — Gamma + Laplace training via mean-rate `beta` gradients.
2. **#1146** (Raashish) — fixed `predictive_values` sampling fallback kwargs for likelihoods without `conditional_mean`; Gamma still failed (`NotImplementedError` — no `samples` either).
3. **#1147** (Neil) — implements `conditional_mean`, `conditional_variance`, and `samples` so `predict` works.

Under the mean-rate form: \(\mathrm{E}[y|f]=\mathrm{link}(f)\), \(\mathrm{Var}[y|f]=\mathrm{link}(f)/\beta\).

## Acceptance Criteria

- [x] `Gamma.conditional_mean` / `conditional_variance` consistent with mean-rate parameterization
- [x] `samples` draws `Gamma(shape=beta*link(f), scale=1/beta)`
- [x] Unit tests for moments, MC mean/var, and end-to-end `predict`
- [x] #1147 merged to `devel`

## Related

- CIP: 0006
- Originating contributor (Laplace/gradients): Raashish Aggarwal (@raashish1601)
- Moments PR author: Neil Lawrence (@lawrennd)
- PRs: #1125, #1146, #1147

## Progress Updates

### 2026-10-10

- #1147 already merged; marking Completed.
