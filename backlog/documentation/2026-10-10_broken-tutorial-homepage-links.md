---
id: 2026-10-10_broken-tutorial-homepage-links
title: Fix broken tutorial and PyPI homepage links
status: Proposed
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: documentation
related_cips:
- '0003'
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

Two high-visibility link failures from open-issue triage (2026-10-10):

1. [#899](https://github.com/SheffieldML/GPy/issues/899) —
   `http://sheffieldml.github.io/GPy/` tutorial entry point redirects poorly /
   fails for newcomers.
2. [#979](https://github.com/SheffieldML/GPy/issues/979) — PyPI project homepage
   URL from packaging metadata 404s.

Metadata URL ownership overlaps [CIP-0003](../../cip/cip0003.md); the tutorial
redirect can land independently via README / docs / GitHub Pages.

## Acceptance Criteria

- [ ] Working canonical tutorial URL documented in README and packaging metadata
- [ ] PyPI homepage / project URLs resolve (no 404)
- [ ] Spot-check `sheffieldml.github.io/GPy` vs nbviewer / notebook repo links
- [ ] Comment / close #899 and #979

## Implementation Notes

- Prefer https and a single canonical docs entry.
- Coordinate metadata edits with CIP-0003 if Poetry/`pyproject.toml` lands;
  otherwise a small `setup.py` / `pyproject.toml` URL fix is enough.

## Related

- CIP: 0003 (metadata)
- Issues: #899, #979
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.
