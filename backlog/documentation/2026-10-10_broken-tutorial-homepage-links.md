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

Two high-visibility link failures from open-issue triage (2026-10-10):

1. [#899](https://github.com/SheffieldML/GPy/issues/899) —
   `http://sheffieldml.github.io/GPy/` / old nbviewer entry points failed for
   newcomers.
2. [#979](https://github.com/SheffieldML/GPy/issues/979) — PyPI project homepage
   URL from packaging metadata was unreliable / 404'd historically.

Metadata URL ownership overlaps [CIP-0003](../../cip/cip0003.md); the tutorial
and project URL fix lands independently via README / `setup.py`.

## Acceptance Criteria

- [x] Working canonical tutorial URL documented in README and packaging metadata
- [x] PyPI homepage / project URLs resolve (no 404)
- [x] Spot-check `sheffieldml.github.io/GPy` vs nbviewer / notebook repo links
- [x] Comment / close #899 and #979

## Implementation Notes

- Canonical set: GitHub repo as `setup.py` `url` (stable for PyPI); Homepage /
  Documentation / Tutorials under `project_urls`.
- README uses `https://` and `nbviewer.org` (not `nbviewer.ipython.org`).
- Follow-up (optional): update the GitHub Pages site tutorial button from
  `nbviewer.ipython.org` to `nbviewer.org` on the `gh-pages` / deploy site
  sources (outside this tree).

## Related

- CIP: 0003 (metadata), 0007 (triage)
- Issues: #899, #979
- Open-issue triage: 2026-10-10

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

- Verified `https://sheffieldml.github.io/GPy/`, Read the Docs deploy docs, and
  nbviewer.org tutorial index return HTTP 200.
- Updated README + `setup.py` project URLs; close #899 / #979 when the PR merges.
