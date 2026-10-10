---
id: 2026-10-10_symmetric-gradients-x-diag
title: Add Symmetric.gradients_X_diag
status: In Progress
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0004'
owner: Neil Lawrence
contributor: mirjanic (@mirjanic)
dependencies: []
tags:
- backlog
- kernels
- gradients
- symmetric
---

# Task: Symmetric `gradients_X_diag`

## Description

Under [CIP-0004](../../cip/cip0004.md) cluster C (kernel correctness), land the
idea from [#1002](https://github.com/SheffieldML/GPy/pull/1002) (@mirjanic).
`Symmetric` implemented `gradients_X` but not `gradients_X_diag`, so diagonal
input-gradient checks used the generic fallback.

Do **not** merge #1002 as-is:

1. On the contributor branch `gradients_X_diag` was defined at **module** scope
   (not indented under the class), so it would not bind as a method.
2. The proposed formula `(1+s)(gdiag(X)+gdiag(AX)Tᵀ)` omits the cross-term
   gradients required by `Kdiag` (`2 s diag(K(AX,X))`). Re-implement with
   correct indentation, cross terms, and a `checkgrad` regression test.

Orthogonal to #1126 (string identity for `symmetry_type`).

## Acceptance Criteria

- [x] `Symmetric.gradients_X_diag` is a class method consistent with `Kdiag`
- [x] Regression test via `Kern_check_dKdiag_dX` for even/odd
- [ ] Credits #1002 / @mirjanic; close #1002 as superseded when the PR merges

## Related

- CIP: 0004
- Contributor: mirjanic (@mirjanic)
- PRs: #1002 (source idea; supersede)

## Progress Updates

### 2026-10-10

- Confirmed #1002 indentation bug and incomplete formula on contributor fork.
- Backlog created; implementing corrected maintainer PR.
