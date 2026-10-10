---
id: 2026-10-10_kernel-tests-continue-after-failure
title: Kernel gradient checks continue after first failure
status: Completed
priority: Low
created: '2026-10-10'
last_updated: '2026-10-10'
category: infrastructure
related_cips:
- '0004'
owner: Neil Lawrence
contributor: bobturneruk (@bobturneruk)
dependencies: []
tags:
- backlog
- testing
- kernels
- pytest
---

# Task: Kernel tests continue after first failure

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster F (test harness), revive the idea from
[#867](https://github.com/SheffieldML/GPy/pull/867) (Robert Turner / @bobturneruk,
2020). `check_kernel_gradient_functions` in `GPy/testing/test_kernel.py` (formerly
`kernel_tests.py`) stops at the first failed sub-check via `assert` + `return False`,
so later gradient failures are hidden.

Land a clean re-implementation: on failure set `pass_checks = False`, keep running
the remaining checks, and return `pass_checks`. Callers already
`assert check_kernel_gradient_functions(...)`, so the test still fails overall.

Do **not** merge #867 as-is (branch gone; file renamed in the pytest migration).

Optional later work (out of scope for this task): split sub-checks into
pytest-parametrized tests so the runner reports each failure separately (as noted
in review on #867).

## Acceptance Criteria

- [x] `check_kernel_gradient_functions` runs all sub-checks after a failure
- [x] Function still returns `False` (and callers fail) if any sub-check failed
- [x] Credits #867 / @bobturneruk; close #867 as superseded when the PR merges
- [x] Optional follow-up noted: pytest parametrize of kernel sub-checks

## Implementation Notes

- Touch only the failure paths inside `check_kernel_gradient_functions`.
- No behaviour change when all checks pass.

## Related

- CIP: 0004
- Contributor: bobturneruk (@bobturneruk)
- PRs: #1150 (merged); #867 (superseded)

## Progress Updates

### 2026-10-10

- Confirmed early-exit pattern still on `devel` in `test_kernel.py`.
- Backlog created; maintainer PR removes `assert`/`return False` early exits.
- Local `TestKernelGradientContinuous` / RBF kernel tests passed.
- [#1150](https://github.com/SheffieldML/GPy/pull/1150) merged to `devel`; #867 closed as superseded.
