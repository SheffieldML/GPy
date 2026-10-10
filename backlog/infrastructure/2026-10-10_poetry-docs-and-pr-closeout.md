---
id: 2026-10-10_poetry-docs-and-pr-closeout
title: Poetry install docs and packaging PR closeout
status: Proposed
priority: Medium
created: '2026-10-10'
last_updated: '2026-10-10'
category: infrastructure
related_cips:
- '0003'
owner: Neil Lawrence
contributor: Martin Bubel (@MartinBubel)
dependencies:
- 2026-10-10_poetry-core-migration
- 2026-10-10_poetry-ci-apple-silicon
tags:
- backlog
- packaging
- poetry
- documentation
---

# Task: Poetry docs + close related packaging PRs

## Description

Finish the [CIP-0003](../../cip/cip0003.md) Option A community surface:

1. Document **end-user** vs **contributor** install paths (pip wheels vs Poetry).
2. Close packaging PRs under the Accepted plan once the migration has landed.

## Acceptance Criteria

- [ ] README (and packaging metadata URLs if needed) describe:
  - End users: `pip install gpy`
  - Contributors: Poetry install / test / lock refresh
- [ ] Comment on [#1080](https://github.com/SheffieldML/GPy/pull/1080) pointing at
      the superseding PR; **close as superseded after that PR merges**
- [ ] Comment + close [#1000](https://github.com/SheffieldML/GPy/pull/1000) as
      absorbed by CIP-0003 / Poetry metadata
- [ ] Comment + close [#1031](https://github.com/SheffieldML/GPy/pull/1031) as
      absorbed (Cython build-only)
- [ ] CIP-0003 implementation checklist updated; mark Implemented when verified

## Implementation Notes

- Ping Martin on #1080 when the superseding PR is open (Accept already happened).
- Coordinate homepage/tutorial URL fixes with
  `backlog/documentation/2026-10-10_broken-tutorial-homepage-links.md` if
  metadata URLs change.

## Related

- CIP: 0003
- PRs: #1080, #1000, #1031

## Progress Updates

### 2026-10-10

CIP-0003 Accepted Option A; backlog created.
