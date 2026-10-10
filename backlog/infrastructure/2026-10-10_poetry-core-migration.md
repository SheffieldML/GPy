---
id: 2026-10-10_poetry-core-migration
title: Poetry core migration superseding #1080
status: Completed
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: infrastructure
related_cips:
- '0003'
owner: Neil Lawrence
contributor: Martin Bubel (@MartinBubel)
dependencies:
- 2026-10-10_poetry-dependency-policy
tags:
- backlog
- packaging
- poetry
- cython
---

# Task: Poetry core migration (supersede #1080)

## Description

Under [CIP-0003](../../cip/cip0003.md) **Option A**, land Poetry packaging on
current `devel` in a **new maintainer PR**. Do **not** merge
[#1080](https://github.com/SheffieldML/GPy/pull/1080) as-is.

Reuse Martin's portable ideas from #1080:

- Poetry `pyproject.toml` + `poetry.lock`
- `build_extension.py` Cython extension hook (not named `build.py`)
- Contributor install via `poetry install` / `poetry run pytest`

Align with the CIP Accept table (name/`gpy`, current version, no dual SoT).

## Acceptance Criteria

- [x] New PR against `devel` (not a force-push of #1080's stale tip)
- [x] Poetry builds wheels/sdists; Cython extensions import after install
- [x] `setup.py` / `setup.cfg` removed or inert so they cannot diverge
- [x] Credits @MartinBubel / #1080 in CHANGELOG and PR body
- [x] Depends on dependency-policy task decisions encoded in the same or prior PR

## Implementation Notes

- Prefer clean re-implementation on `devel` over fighting a 477+ commit rebase.
- Keep end-user `pip install gpy` working from the built wheel.
- Coordinate CI changes with `2026-10-10_poetry-ci-apple-silicon` (may be same PR
  if scope stays reviewable).

## Related

- CIP: 0003
- Contributor: Martin Bubel (@MartinBubel)
- PRs: #1080 (source; close as superseded after merge)

## Progress Updates

### 2026-10-10

CIP-0003 Accepted Option A; backlog created.

- Implementing on branch `infra/poetry-option-a`.

- Landed via [#1164](https://github.com/SheffieldML/GPy/pull/1164); closed #1080 / #1000 / #1031.
