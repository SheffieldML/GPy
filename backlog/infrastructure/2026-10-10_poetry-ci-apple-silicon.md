---
id: 2026-10-10_poetry-ci-apple-silicon
title: Poetry CI matrix and Apple Silicon green
status: In Progress
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: infrastructure
related_cips:
- '0003'
owner: Neil Lawrence
dependencies:
- 2026-10-10_poetry-core-migration
tags:
- backlog
- packaging
- poetry
- ci
- macos
---

# Task: Poetry CI + Apple Silicon

## Description

Under [CIP-0003](../../cip/cip0003.md) Option A, rewrite test/build workflows to
install and exercise the package via Poetry (or `pip install` from a Poetry-built
wheel), on the current platform matrix including **Apple Silicon**
(`macos-latest`).

#1080's original blocker was M-series CI failure. That is now more relevant, not
less. Do not ship the Poetry migration without green macOS CI **or** a written
deferral recorded in CIP-0003.

## Acceptance Criteria

- [ ] CI uses current Actions (no `checkout@v1` / `upload-artifact@v3` leftovers)
- [ ] Linux / Windows / macOS jobs green for supported Python versions
- [ ] Cython extension import smoke check on each OS
- [ ] Wheel build + `pip install` path covered (end-user story)
- [ ] Apple Silicon green, **or** CIP-0003 updated with explicit deferral + rationale

## Implementation Notes

- Prefer proving green over dropping macOS.
- Keep release/deploy gating consistent with current `devel` (release-only wheels
  if that policy remains).
- May share a PR with the core migration if the diff stays reviewable.

## Related

- CIP: 0003
- PRs: #1080 (Apple Silicon discussion)
- Comment: https://github.com/SheffieldML/GPy/pull/1080#issuecomment-2255203278

## Progress Updates

### 2026-10-10

CIP-0003 Accepted Option A; backlog created.

- Implementing on branch `infra/poetry-option-a`.
