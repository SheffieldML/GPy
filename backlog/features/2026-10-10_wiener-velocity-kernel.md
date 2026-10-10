---
id: 2026-10-10_wiener-velocity-kernel
title: Land Wiener Velocity kernel from #1003
status: Completed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: features
related_cips:
- '0004'
owner: Neil Lawrence
contributor: Duncan Gulla (@Duncan10)
dependencies: []
tags:
- backlog
- kernels
- wiener
- brownian
---

# Task: Wiener Velocity kernel (#1003)

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster C, land
[#1003](https://github.com/SheffieldML/GPy/pull/1003) (@Duncan10): a 1D Wiener
Velocity kernel (once-integrated Brownian motion / Solin 2016), parallel to the
existing `Brownian` kernel.

The contributor implementation is sound. Do **not** rewrite from scratch —
rebase onto current `devel` and relocate the gradient test into
`GPy/testing/test_kernel.py` (the old `kernel_tests.py` path was renamed).

## Acceptance Criteria

- [x] `GPy.kern.WienerVelocity` exported from `GPy.kern`
- [x] `K` / `Kdiag` / `update_gradients_full` + `to_dict` / `from_dict`
- [x] Gradient check in `test_kernel.py` (`test_WienerVelocity`)
- [x] CI green on #1003 after rebase
- [x] CHANGELOG entry; credit @Duncan10

## Implementation Notes

- Head branch is `Duncan10:devel` (`maintainer_can_modify`); update that PR
  in place rather than opening a superseding maintainer PR.
- Only rebase conflict was modify/delete on the renamed test module.
- Keep Duncan's `useGPU` constructor kwarg for load compatibility.

## Related

- CIP: 0004
- Contributor: Duncan Gulla (@Duncan10)
- PRs: #1003

## Progress Updates

### 2026-10-10

- Assessed #1003; backlog created; rebasing original PR onto `devel` with
  test relocated to `test_kernel.py`.
- Rebased in place; CI green; merged #1003.
