---
id: 2026-10-11_reinstate-coverage-ci
title: Reinstate coverage reporting on GitHub Actions
status: Proposed
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
badges still point at Coveralls/Codecov, but GitHub Actions only runs bare
`pytest` with no `--cov` and no upload. Coveralls last reported ~57% in May
2021; the numbers look “dropped” because uploads stopped, not because the suite
collapsed overnight.

Reinstate a **light** coverage path: one Codecov upload from Actions, refresh
badges, and keep status informational until a trustworthy baseline exists.

## Acceptance Criteria

- [ ] `pytest-cov` (or equivalent) available in the Poetry `dev` group
- [ ] One GHA job (single OS/Python) runs `pytest --cov=GPy --cov-report=xml`
      and uploads to Codecov (or Coveralls — pick one)
- [ ] README status table updated: drop dead Travis/AppVeyor badges; show live
      coverage badge for `devel`
- [ ] Coverage status starts informational (no hard fail on % drop), or uses a
      soft `target: auto` with documented rationale
- [ ] Review `.coveragerc` excludes (`raise` / `except` / `pass` currently
      inflate reported coverage) and document intentional omissions

## Implementation Notes

- Prefer **Codecov + Actions** over restoring the dual Coveralls+Codecov+Travis
  stack.
- Do not run coverage on the full OS/Python matrix initially (cost vs signal).
- Local HTML reports (`coverage html`) can stay documented for contributors.
- Related motivation: SIM115 / file-IO work (#1202) showed how little of
  `datasets`/`mocap` was covered; a live report makes that visible on PRs.

## Related

- CIP: 0005 (developer tooling / CI quality)
- Config leftovers: `.coveragerc`, `codecov.yml`
- PR context: #1202 (file-IO tests ahead of SIM115)
- Historical: Coveralls build ~4087 (2021-05-13), ~57%

## Progress Updates

### 2026-10-11

Backlog created after noting stale README badges and no coverage upload on the
current `test-and-deploy.yml` workflow.
