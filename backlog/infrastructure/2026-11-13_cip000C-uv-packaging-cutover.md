---
id: 2026-11-13_cip000C-uv-packaging-cutover
title: CIP-000C — migrate contributor packaging from Poetry to uv
status: In Progress
priority: High
created: '2026-11-13'
last_updated: '2026-11-13'
category: infrastructure
related_cips:
- '000C'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- packaging
- uv
- poetry
- ci
- cip000C
---

# Task: CIP-000C uv packaging cutover

## Description

Replace Poetry as the contributor/CI install and lock tool with uv, keeping
`poetry-core` + `build_extension.py` for Cython in this cutover. End-user
`pip install gpy` stays unchanged.

## Acceptance Criteria

- [x] `pyproject.toml` uses PEP 621 + `[dependency-groups]`; build backend still
      `poetry-core` with `build_extension.py`
- [x] `uv.lock` committed; `poetry.lock` removed
- [x] `uv sync` + Cython smoke import + `pytest GPy/testing` green locally
- [x] CI workflows use uv (`astral-sh/setup-uv` or equivalent); no `poetry install`
- [x] README contributor path documents `uv sync` / `uv run`
- [x] CIP-0008 tag check reads PEP 621 `[project].version`
- [ ] Dependabot still viable for the lockfile (or scheduled `uv lock --upgrade`)

## Related

- CIP: 000C
- Discussion: [#1268](https://github.com/SheffieldML/GPy/discussions/1268)
- Branch: `cip000C-uv-packaging`

## Progress Updates

### 2026-11-13

CIP Accepted with phased decisions. Worktree on `cip000C-uv-packaging`.

`migrate-to-uv --keep-current-build-backend --ignore-locked-versions` plus
manual restore of `[tool.poetry.build]` (tool dropped the Cython hook). Local
spike green: Cython smoke, `uv build` wheel import, 410 pytest passed / 21
skipped. CI + README cut over; awaiting PR CI and Dependabot check.
