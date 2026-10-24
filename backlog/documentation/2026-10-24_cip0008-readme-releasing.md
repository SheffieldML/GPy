---
id: 2026-10-24_cip0008-readme-releasing
title: CIP-0008 — README releasing instructions
status: Completed
priority: Medium
created: "2026-10-24"
last_updated: "2026-10-24"
category: documentation
related_cips:
- "0008"
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- documentation
- release
---

# Task: README how to cut a GPy release

## Description

Replace the stale “PR `devel` → `deploy`” / bumpversion / travis_tests story
with the actual loop: merge to the default branch, bump Poetry version,
publish a GitHub Release `vX.Y.Z`, watch `Test Python Lib` deploy.

## Acceptance Criteria

- [x] README “Deploying GPy” matches CIP-0008 (Release `types: published`)
- [x] Contributor PR target is the default branch, not a hard-coded
      `deploy` merge
- [x] No instruction to open `devel` → `deploy` to upload to PyPI

## Related

- CIP: 0008

## Progress Updates

### 2026-10-24

Task created In Progress with the CI change.

README releasing steps now say merge to `main` and publish a GitHub
Release. Contributor/clone/badge/RTD links follow the default-branch
rename.
