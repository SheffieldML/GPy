---
id: 2026-10-10_link-parameter-leftover-kernels
title: Fix leftover add_parameter → link_parameter in kernels
status: Completed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0004'
owner: Neil Lawrence
contributor: tallakahath (@tallakahath)
dependencies: []
tags:
- backlog
- kernels
- paramz
---

# Task: Leftover `add_parameter` → `link_parameter`

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster C (kernel correctness), land
[#978](https://github.com/SheffieldML/GPy/pull/978) (Liz Decolvenaere /
@tallakahath). After the paramz rename, `splitKern` / `DEtime` and
`TruncLinear` / `TruncLinear_inf` still called `add_parameter`.

Author refreshed the PR onto modern `devel` (with regression tests) after the
CIP-0004 invite; maintainers merged their branch rather than re-implementing.

## Acceptance Criteria

- [x] `link_parameter` used in `splitKern.py` and `trunclinear.py`
- [x] Regression tests for TruncLinear / DEtime parameter linking
- [x] #978 merged; author credited

## Related

- CIP: 0004
- Contributor: tallakahath (@tallakahath)
- PRs: #978 (merged)

## Progress Updates

### 2026-10-10

- CIP-0004 invite; author refreshed and confirmed ready.
- [#978](https://github.com/SheffieldML/GPy/pull/978) merged to `devel`.
