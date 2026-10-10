---
id: 2026-10-10_broken-tutorial-homepage-links
title: Fix broken tutorial and PyPI homepage links
status: Completed
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: documentation
related_cips:
- '0003'
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- documentation
- links
- pypi
---

# Task: Broken tutorial / homepage links

## Description

High-visibility link failures from open-issue triage: tutorial entry (#899) and
PyPI homepage metadata (#979).

## Acceptance Criteria

- [x] Working canonical tutorial URL documented in README and packaging metadata
- [x] PyPI homepage / project URLs resolve (no 404)
- [x] Spot-check `sheffieldml.github.io/GPy` vs nbviewer / notebook repo links
- [x] Comment / close #899 and #979

## Implementation Notes

Landed with the CIP-0007 High backlog / links slice (#1161 and related). CIP-0003
Poetry packaging also carries project URLs.

## Related

- Issues: #899, #979 (closed)
- CIP: 0003, 0007

## Progress Updates

### 2026-10-10

Task created; completed during High backlog execution.
