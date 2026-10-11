---
id: 2026-10-11_retire-user-mailing-list
title: Retire Sheffield gpy-users mailing list links; point to GitHub Discussions
status: Completed
priority: Medium
created: '2026-10-11'
last_updated: '2026-10-11'
category: documentation
related_cips:
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- documentation
- mailing-list
- discussions
---

# Task: Retire user mailing list documentation

## Description

Discussion [#934](https://github.com/SheffieldML/GPy/discussions/934): the
Sheffield Sympa list
(`https://lists.shef.ac.uk/sympa/subscribe/gpy-users`) no longer works.
README and the GitHub Pages homepage still advertised it.

Community Q&A now lives on GitHub Discussions for this repository.

## Acceptance Criteria

- [x] Remove dead mailing-list link from `README.md`; point to GitHub Discussions
- [x] Remove mailing-list section from `gh-pages` `index.html`; point to Discussions
- [x] No remaining live docs links to `lists.shef.ac.uk` / `gpy-users` (CHANGELOG history OK)
- [x] Note resolution on Discussion #934

## Implementation Notes

- README bullet: GitHub Discussions URL
- Homepage (`sheffieldml.github.io/GPy`): replace “User mailing list” with
  “Community / Discussions”

## Related

- Discussion: #934
- Issue: #1219
- PR: #1218
- CIP: 0007

## Progress Updates

### 2026-10-11

Task created and completed in the same change. Converted Discussion #934 →
Issue #1219 for PR tracking (same pattern as #958 → #1217).
