---
author: Neil Lawrence
created: "2026-11-10"
id: "000C"
last_updated: "2026-11-13"
status: In Progress
compressed: false
related_requirements: []
related_cips:
- "0003"
- "0008"
- "000A"
tags:
- cip
- packaging
- uv
- poetry
- build-system
- infrastructure
- ci
title: Contributor packaging via uv (succeed Poetry)
---

# CIP-000C: Contributor packaging via uv (succeed Poetry)

## Status

- [x] Proposed - Initial idea documented (open for Discussion feedback)
- [x] Accepted - Approved, ready to start work (Accept decisions below)
- [x] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Replace **Poetry** as GPy's contributor and CI install/lock tool with
[**uv**](https://docs.astral.sh/uv/), while keeping the CIP-0003 end-user
contract (`pip install gpy` from wheels) and the Cython extension build.

CIP-0003 closed with Poetry as the contributor source of truth ([#1164](https://github.com/SheffieldML/GPy/pull/1164),
superseding [#1080](https://github.com/SheffieldML/GPy/pull/1080)). That choice
solved packaging drift; it did not commit us to Poetry forever. Local and CI
installs are now the slow part of the loop. uv is substantially faster for
resolve and install, which matters on every PR matrix cell and every clone.

Community review: [Discussion #1268](https://github.com/SheffieldML/GPy/discussions/1268).

## Motivation

- **Install latency.** Contributor and CI paths today are
  `pip install poetry==…` then `poetry install --with dev`. That dominates cold
  PR jobs more than pytest does. uv installs the same locked graph much faster
  (resolve + fetch + venv), which shortens feedback and lowers Actions minutes.
- **One modern toolchain.** Contributors already use Ruff (CIP-0005). uv is
  from the same Astral stack and is becoming the default for locked Python
  projects.
- **CIP-0003 goals still stand.** Reproducible contributor installs, NumPy ≥ 2,
  Cython build-only, optional `tables`, Apple Silicon first-class, end users on
  wheels. Only the *tool* that owns the lockfile and editable install changes.
- **End users already win with uv optionally.** `uv pip install gpy` is faster
  than `pip install gpy` without any GPy change; this CIP is about *our* repo
  workflow and CI, not forcing end users onto uv.

## Detailed Description

### What stays (from CIP-0003 / CIP-0008)

| Topic | Policy |
|-------|--------|
| End users | `pip install gpy` (wheels); uv not required at runtime |
| Distribution name / version | `gpy` / Poetry or PEP 621 version SoT in `pyproject.toml` |
| NumPy / SciPy / Cython / `tables` | Unchanged Accept table from CIP-0003 |
| Release trigger | GitHub Release `vX.Y.Z` → wheels → PyPI (CIP-0008) |
| Cython extensions | Still built from `build_extension.py` (or a successor hook) |

### What changes

| Area | Today (Poetry) | Target (uv) |
|------|----------------|-------------|
| Contributor install | `poetry install --with dev` | `uv sync --group dev` (or equivalent) |
| Run tests | `poetry run pytest …` | `uv run pytest …` |
| Lockfile | `poetry.lock` | `uv.lock` (commit; document that PyPI installs ignore it) |
| CI bootstrap | `pip install poetry==2.1.3` | install uv (official action or `curl` installer); drop Poetry from the matrix |
| Build wheels | `poetry build` | `uv build` (or `python -m build` once the backend is non-Poetry) |
| Version check in deploy | `poetry version -s` | read PEP 621 / remaining Poetry version field consistently |

### Hard constraint: Cython build hook

Poetry owns compilation today via:

```toml
[tool.poetry.build]
script = "build_extension.py"
generate-setup-file = true
```

and `build_extension.build(setup_kwargs)` injecting `ext_modules`. That API is
**Poetry-specific**. A naive `migrate-to-uv` that only swaps the lockfile will
break wheel builds unless we keep a working PEP 517 path for Cython.

**Preferred approach (phased):**

1. **Lock + sync first.** Move contributor/CI install to uv. Prefer PEP 621
   `[project]` metadata where it is already natural. Keep a build backend that
   still compiles extensions (initially may still be `poetry-core` with
   `--keep-current-build-backend`, or an explicit setuptools/`build` path that
   calls the same Extension list).
2. **Retire Poetry commands from CI and README.** No `poetry install` /
   `poetry run` in workflows or contributor docs.
3. **Normalize the build backend** so Cython does not depend on Poetry's
   `build()` hook forever (setuptools `ext_modules` via a small `build.py` /
   hatchling plugin / retained poetry-core only as a build-system dependency).
   Document the chosen backend in this CIP before Accept closes open questions.
4. **Remove `poetry.lock`** once `uv.lock` is the sole committed lockfile and
   CI is green on Linux / Windows / macOS (incl. arm64).

**Non-goals for this CIP**

- Changing the public import API or Cython extension set.
- Mandatory TestPyPI.
- Forcing end users to install uv.
- Rewriting CIP-000A's PR matrix shape (uv may make a fuller PR matrix cheaper;
  that is a follow-on, not a requirement here).

### Alternatives considered

| Option | Verdict |
|--------|---------|
| **A — uv for lock/sync/CI; keep end-user pip** (this CIP) | Preferred |
| B — Stay on Poetry; tune CI caching only | Rejects the observed install-time win |
| C — Dual Poetry + uv lockfiles indefinitely | Transition aid only; do not leave both as SoT |
| D — Drop lockfiles entirely | Conflicts with CIP-0003 contributor reproducibility |

## Implementation Plan

1. **Proposed (this phase):** document design; open Ideas Discussion; gather
   feedback on build-backend choice and command mapping.
2. **Spike (after Accept):** prove on a branch that `uv sync` + Cython smoke
   import + `pytest GPy/testing` works on at least Ubuntu and macOS arm64;
   prove `uv build` (or chosen builder) produces importable wheels.
3. **Metadata / lock:** migrate `pyproject.toml` toward PEP 621 as needed;
   generate and commit `uv.lock`; stop committing `poetry.lock`.
4. **CI:** replace Poetry bootstrap in `.github/workflows/test-and-deploy.yml`
   (and any Dependabot poetry ecosystem config) with uv; keep CIP-0008 release
   tag check against the same version SoT.
5. **Docs:** README contributor section → uv; note optional `uv pip install gpy`
   for end users who already have uv; update CIP-0008 / CIP-000A references that
   say `poetry version -s` / `poetry install`.
6. **Closeout:** Dependabot / security refresh path uses `uv.lock`; close
   Discussion; mark CIP Closed; compress into README when ready.

## Backward Compatibility

- **`pip install gpy`:** unchanged.
- **Contributors:** must install uv instead of Poetry. Document the one-liner
  (`curl` installer or package manager). Editable/dev workflow commands change
  (`uv sync` / `uv run`).
- **Lockfile:** `poetry.lock` consumers switch to `uv.lock`; do not keep both
  after cutover.
- **Build from sdist:** must still compile Cython when no wheel matches (same
  as today).

## Testing Strategy

- Cold and warm `uv sync --group dev` vs current `poetry install` timing on a
  maintainer machine (record numbers in the Discussion or CIP update).
- CI matrix: Linux, Windows, macOS; Python versions supported by current policy
  (≥3.11).
- Smoke-import all Cython extensions after sync and after wheel install.
- `uv build` (or chosen builder) → `pip install dist/gpy-*.whl` → import +
  subset pytest.
- Release dry-run (`workflow_dispatch`) still produces manylinux x86_64 +
  aarch64, macOS, Windows wheels.
- Dependabot or equivalent can open PRs that refresh `uv.lock`.

## Related Requirements

None yet. Candidates after Accept:

- Fast, reproducible contributor installs
- CI install time bounded relative to test time
- Single lockfile SoT for development environments

## Accept decisions (2026-11-13)

Resolved with no Discussion replies after three days; proceed with the
preferred phased cutover.

1. **Build backend:** keep `poetry-core` + `build_extension.py` for this
   cutover (`migrate-to-uv --keep-current-build-backend`). Move Cython off
   Poetry’s `build()` hook in a follow-on PR after CI is green on `uv.lock`.
2. **Dependency groups:** Poetry `group.dev` → PEP 735 `[dependency-groups]
   dev`; Poetry extras → `[project.optional-dependencies]`. Contributor
   default remains `uv sync` with `dev` (uv’s default group).
3. **Dependabot:** keep `package-ecosystem: pip` on `/` for now (reads
   `pyproject.toml` / lock); verify PRs still refresh after `uv.lock` lands.
   If Dependabot cannot update `uv.lock`, add a scheduled `uv lock --upgrade`
   workflow in closeout.
4. **Version SoT:** single PEP 621 `[project].version`; CIP-0008 tag check
   reads that field (not `tool.poetry.version`).

Command map for docs/CI: `poetry install --with dev` → `uv sync`;
`poetry run …` → `uv run …`; `poetry build` → `uv build`.

## Implementation Status

- [x] Draft CIP-000C (Proposed)
- [x] Ideas Discussion opened: [#1268](https://github.com/SheffieldML/GPy/discussions/1268)
- [x] Accept decisions: build backend, command map, Dependabot
- [x] Spike: uv sync + Cython + pytest green (410 passed, 21 skipped; wheel import ok)
- [x] Backlog tasks created (only after Accept)
- [x] CI and README cut over; `poetry.lock` removed
- [ ] First release built with uv-based CI verified; CIP Closed

## References

- Predecessor: [CIP-0003](cip0003.md) (Poetry; Closed via [#1164](https://github.com/SheffieldML/GPy/pull/1164))
- Original Poetry PR: [#1080](https://github.com/SheffieldML/GPy/pull/1080) (superseded)
- Release pipeline: [CIP-0008](cip0008.md); PR CI shape: [CIP-000A](cip000A.md)
- uv docs: https://docs.astral.sh/uv/
- Poetry → uv migration notes: https://docs.astral.sh/uv/guides/migration/poetry/
- Discussion: [#1268](https://github.com/SheffieldML/GPy/discussions/1268)
