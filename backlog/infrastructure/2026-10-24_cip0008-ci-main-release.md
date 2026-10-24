---
id: 2026-10-24_cip0008-ci-main-release
title: CIP-0008 CI — published-only release, wheel dry-run, drop deploy trigger
status: Completed
priority: High
created: "2026-10-24"
last_updated: "2026-10-24"
category: infrastructure
related_cips:
- "0008"
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- infrastructure
- ci
- release
- packaging
---

# Task: CIP-0008 CI (main-centric publish)

## Description

Implement the CI half of CIP-0008 now that GPy 1.15.0 is on PyPI:

- `on.release.types: [published]` only (v1.15.0 queued three workflow copies)
- Wheel jobs on `workflow_dispatch` without PyPI upload
- Drop `deploy` from `push` triggers; keep `main` and `devel` until the
  default branch is renamed
- Fail deploy unless the Release tag is `v$(poetry version -s)`
- Restrict `twine upload` to `*.whl` / `*.tar.gz` (1.14.2 `No PKG-INFO`)

Do not change Poetry/Cython build mechanics. Keep `twine` + `id-token`
(1.15.0 already uploaded that way).

## Acceptance Criteria

- [x] `Test Python Lib` does not start wheel/deploy jobs on ordinary PRs
- [x] `workflow_dispatch` builds Windows/macOS/manylinux (incl. aarch64)
      artifacts and does not run `twine upload`
- [x] Publishing a GitHub Release (`types: published`) still tests, builds,
      and uploads
- [x] Tag `vX.Y.Z` must match `pyproject.toml` Poetry version
- [x] `deploy` branch is not a CI trigger

## Related

- CIP: 0008
- Discussion: [#1228](https://github.com/SheffieldML/GPy/discussions/1228)
- Release: [v1.15.0](https://github.com/SheffieldML/GPy/releases/tag/v1.15.0)

## Progress Updates

### 2026-10-24

Task created In Progress. 1.15.0 is on PyPI (`requires_python >=3.9,<4`,
aarch64 wheels present).

Landed in [#1244](https://github.com/SheffieldML/GPy/pull/1244). Default
branch is now `main`; a `workflow_dispatch` dry-run was queued on that
branch.
