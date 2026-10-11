---
id: 2026-10-11_reinstate-coverage-ci
title: Reinstate coverage reporting on GitHub Actions
status: Completed
priority: Medium
created: '2026-10-11'
last_updated: '2026-10-11'
category: infrastructure
related_cips:
- '0005'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- ci
- coverage
- codecov
- testing
---

# Task: Reinstate coverage CI

## Description

Coverage monitoring for GPy stopped when CI moved off Travis/AppVeyor. README
badges still pointed at Coveralls/Codecov, but GitHub Actions only ran bare
`pytest` with no `--cov` and no upload.

Reinstate a **light** coverage path: one Codecov upload from Actions, refresh
badges, and keep status informational until a trustworthy baseline exists.

## Acceptance Criteria

- [x] `pytest-cov` (or equivalent) available in the Poetry `dev` group
- [x] One GHA job (single OS/Python) runs `pytest --cov=GPy --cov-report=xml`
      and uploads to Codecov (or Coveralls — pick one)
- [x] README status table updated: drop dead Travis/AppVeyor badges; show live
      coverage badge for `devel`
- [x] Coverage status starts informational (no hard fail on % drop), or uses a
      soft `target: auto` with documented rationale
- [x] Review `.coveragerc` excludes (`raise` / `except` / `pass` currently
      inflate reported coverage) and document intentional omissions

## Implementation Notes

- Prefer **Codecov + Actions** over restoring the dual Coveralls+Codecov+Travis
  stack.
- Do not run coverage on the full OS/Python matrix initially (cost vs signal).
- Repo secret `CODECOV_TOKEN` (repository upload token UUID) required for
  uploads from protected branches.

## Related

- CIP: 0005 (developer tooling / CI quality)
- PR: [#1205](https://github.com/SheffieldML/GPy/pull/1205)
- Example successful upload:
  https://app.codecov.io/github/SheffieldML/GPy/commit/478ae9af41ad3fdd7eb17aa5de8692f21a05a17d

## Progress Updates

### 2026-10-11

Backlog created after noting stale README badges and no coverage upload on the
current `test-and-deploy.yml` workflow.

Merged [#1205](https://github.com/SheffieldML/GPy/pull/1205); Codecov upload
verified on the PR commit. Marked **Completed**.
