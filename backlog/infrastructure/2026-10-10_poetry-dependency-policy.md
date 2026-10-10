---
id: 2026-10-10_poetry-dependency-policy
title: Encode Poetry dependency policy (NumPy 2, SciPy build, Cython, tables)
status: In Progress
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: infrastructure
related_cips:
- '0003'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- packaging
- poetry
- numpy
- cython
---

# Task: Poetry dependency policy

## Description

Encode the [CIP-0003](../../cip/cip0003.md) Accept-table dependency decisions in
`pyproject.toml` / `poetry.lock` (and any build hook config):

| Topic | Policy |
|-------|--------|
| NumPy | `>=2` (match `devel`; do not revive #1080's `<2`) |
| SciPy | In `[build-system].requires` for `cython_blas` cimports |
| Cython | Build-time only ([#1031](https://github.com/SheffieldML/GPy/pull/1031)) |
| `tables` | Optional extra, not hard core |
| Lockfile | Commit `poetry.lock` for contributors; document PyPI ignores it |
| Python | Match current `devel` CI matrix (3.11–3.14 unless CIP updates) |

Absorbs the intent of #1000 (machine-readable pep508 ranges) without merging
that PR as-is.

## Acceptance Criteria

- [ ] `pyproject.toml` reflects the Accept table
- [ ] Isolated PEP 517 build succeeds with SciPy present
- [ ] Cython not required to *import* a released wheel
- [ ] `tables` only via extra
- [ ] Fresh `poetry.lock` committed and regenerable
- [ ] Note in PR how #1000 / #1031 are absorbed

## Implementation Notes

- Can land in the same PR as `poetry-core-migration` if clearer as one review.
- Verify paramz on NumPy 2 before tightening ranges further.

## Related

- CIP: 0003
- PRs: #1080 (context), #1000, #1031

## Progress Updates

### 2026-10-10

CIP-0003 Accepted Option A; backlog created.

- Implementing on branch `infra/poetry-option-a`.
