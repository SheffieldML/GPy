---
author: Neil Lawrence
created: "2026-10-25"
id: "000A"
last_updated: "2026-10-25"
status: Proposed
compressed: false
related_requirements: []
related_cips:
- "0005"
- "0008"
- "0009"
tags:
- cip
- ci
- github-actions
- testing
- community
title: Tiered GitHub Actions for pull requests vs publish
---

# CIP-000A: Tiered GitHub Actions for pull requests vs publish

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Cut GitHub Actions cost and queue time on pull requests. Today every PR
runs the full **3 OS × 4 Python** pytest matrix (12 Cython-compile jobs)
plus lint and coverage — including docs-only and backlog-only PRs.

Keep the **full matrix for publish** (GitHub Release `published`) and for
an explicit dry-run. On PRs: skip tests when no code changed; otherwise
run a **canary** (Ubuntu / Python 3.11, the declared floor) plus lint.
Do not drop Windows / 3.12–3.14 coverage; move it off the default PR path.

This CIP is process/infra HOW. It revises CIP-0008's "push / PR: test +
lint (every commit)" line without changing the release-publish shape.

## Motivation

GPy's test job compiles Cython extensions on every matrix cell. A
documentation or CIP PR (for example #1247, three markdown files) still
pays for 12 compile-and-pytest jobs. That slows review, burns Actions
minutes, and makes it expensive to keep PRs small.

We still need the full matrix before wheels go to PyPI. We do not need it
to decide whether a README sentence is mergeable.

## Detailed Description

### Current shape (after CIP-0008)

`.github/workflows/test-and-deploy.yml`:

| Event | `test` matrix | `lint` / `coverage` | wheels |
|-------|---------------|---------------------|--------|
| `pull_request` (any path) | 3 OS × 3.11–3.14 (12) | yes | no |
| `push` to `main` | same 12 | yes | no |
| `workflow_dispatch` | same 12 | yes | yes (no PyPI) |
| `release` `published` | same 12 | yes | yes → PyPI |

`pull_request:` has no path filters and no concurrency cancel.

### Goals

1. **Docs-only PRs are cheap.** Changes under `cip/`, `backlog/`,
   `tenets/`, `docs/`, and `*.md` (with no `GPy/`, packaging, or workflow
   files) do not compile Cython or run pytest.
2. **Code PRs get a canary.** Default: Ubuntu + Python **3.11** + ruff.
   That is the package floor (`python = ">=3.11,<4"`).
3. **Publish stays thorough.** Release `published` (and
   `workflow_dispatch` dry-run) keep today's 3 × 4 test matrix and the
   wheel jobs.
4. **Branch protection still works.** Skipping jobs must not leave
   required checks "pending". One always-run **CI gate** job is the
   required check.
5. **No behaviour change for `pip install gpy`.** Test policy only.

### Non-goals

- Changing Poetry / Cython build mechanics (CIP-0003).
- Changing Codecov *policy* (CIP-0009); only *when* the coverage job runs.
- Reworking Trusted Publishing / tag checks (CIP-0008).
- Dropping 3.12–3.14 or Windows/macOS from supported CI; they move to
  publish (and optional extra triggers).

### Recommended target shape

```text
  pull_request
           │
           ├─ always: concurrency cancel-in-progress; paths-filter; ci-gate
           ├─ docs-only: ci-gate succeeds; no test / coverage / lint
           └─ code: lint + test (ubuntu-latest, 3.11)
                    coverage if GPy/ changed (ubuntu, 3.12, CIP-0009)

  push to main
           └─ same as a code PR (canary), not the 12-cell matrix
              (see open question 1)

  workflow_dispatch / release published
           └─ full test matrix + wheels (unchanged CIP-0008)
```

**Why 3.11 on PRs, not 3.13.** The floor is what contributors and old
environments hit first. Newer CPython is more likely to show up as a
wheel-build failure, which publish already covers.

### Docs-only vs code

Treat a change as **code** if any path matches:

- `GPy/`
- `pyproject.toml`, `poetry.lock`, `build_extension.py`
- `.github/workflows/`
- `*.pyx`, `*.pxd`, `*.c`, `*.h`

Treat as **docs-only** if every path is under `cip/`, `backlog/`,
`tenets/`, `docs/`, or is markdown / licence / authors at the repo root.

A mixed PR is **code**.

### Cython / Windows risk

Ubuntu 3.11 will not catch MSVC or macOS compile breaks. Mitigations,
cheapest first:

1. If the PR touches `*.pyx` / `*.pxd` / `build_extension.py`, add
   **Windows + 3.11** to the PR matrix (one extra cell).
2. Label **`full-ci`** forces the 12-cell matrix on that PR.
3. Optional **weekly cron** (`workflow_dispatch`-style, no wheels) runs
   the full test matrix on `main` so a silent Windows drift is not
   deferred until the next release.

### Required-checks trap

GitHub `paths-ignore` at workflow level skips the *entire* workflow.
If branch protection requires `test (ubuntu-latest, 3.11)` and that job
never starts, the PR cannot merge.

Do **not** use workflow-level `paths-ignore` as the only skip. Use
job-level `if:` from a `paths-filter` (for example
[dorny/paths-filter](https://github.com/dorny/paths-filter)) plus a
`ci-gate` job:

```text
ci-gate needs: [test, lint, coverage]
  if: always()
  succeed when skipped-or-green; fail when any needed job failed
```

Branch protection then requires **`ci-gate` only**, not each matrix
cell. CIP-0008 rename work should land this name once; do not keep
requiring the old 12 check names.

### Concurrency (do this regardless)

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

Cancel superseded PR runs. Do not cancel `release` or `workflow_dispatch`.

### Alternatives considered

| Approach | Verdict |
|----------|---------|
| Workflow `paths-ignore` only | Reject as sole mechanism (required-checks trap). |
| PR = Ubuntu 3.11; **full matrix only at publish**; no cron, no `full-ci` | Allowed if we accept Windows/macOS/newer-Python bugs surfacing at release. Record as open question 1–3. |
| Split `ci-pr.yml` and `test-and-deploy.yml` | Optional later. One workflow with job `if:` is enough for this CIP. |
| Merge queue with full matrix | Heavier than we need; not this CIP. |

### Decisions (open questions)

1. **`push` to `main`:** **canary** (same as PR) vs **full 12-cell matrix**.
   Recommendation: **canary**, plus optional weekly full matrix (question 3).
2. **PR canary OS:** Ubuntu 3.11 only vs Ubuntu **and** Windows 3.11 always.
   Recommendation: Ubuntu always; add Windows only when Cython/build files
   change.
3. **Weekly full matrix on `main`:** yes / no.
   Recommendation: **yes** (pytest only, no wheels) so publish is not the
   first time we see 3.14-on-Windows.
4. **`full-ci` label:** yes / no. Recommendation: **yes**.
5. **Coverage job on PRs:** keep Ubuntu 3.12 when `GPy/` changes (CIP-0009
   patch reports); skip on docs-only.

## Implementation Plan

1. **Accept this CIP** with the open questions resolved.
2. **Workflow:** `concurrency`; `paths-filter`; job `if:` for
   test / lint / coverage; `ci-gate`; slim PR/`push` matrix; keep full
   matrix on `release` and `workflow_dispatch`.
3. **Branch protection:** required check = `ci-gate`. Drop per-cell
   `test (os, python)` requirements if any are set.
4. **Label** `full-ci` (and a one-line README / contributing note after
   CIP-0008 compression, or here if README is the live release doc).
5. **Optional cron** job if question 3 is yes.
6. **Verify** with three PRs: docs-only, Python-only, Cython-touching.

Coordinate with CIP-0008: same YAML file. Prefer landing this *after*
the `devel` → `main` rename PR so we do not fight two workflow diffs.
If 0008 is still open, this CIP can wait or stack on that branch.

## Backward Compatibility

- End users: none.
- Contributors: faster green on docs/CIP/backlog PRs; code PRs no longer
  wait on 12 cells. Windows/macOS/3.12+ failures may appear later
  (publish, weekly, or `full-ci`) unless we take the stricter options
  above.
- Maintainers: one required check to configure; `full-ci` label to opt in.

## Testing Strategy

- Docs-only PR: `test` / `coverage` skipped; `ci-gate` green; no Cython
  compile in the logs.
- `GPy/` Python-only PR: one Ubuntu 3.11 pytest job + lint (+ coverage).
- `*.pyx` PR: Ubuntu 3.11 + Windows 3.11 (if question 2 is accepted).
- `full-ci` labelled PR: 12-cell matrix.
- `workflow_dispatch` / dummy Release path: full matrix + wheels unchanged.
- Confirm a skipped `test` job does not block merge.

## Related Requirements

None yet. Infra HOW for contributor cost after CIP-0008.

## Implementation Status

- [x] Draft CIP-000A (Proposed)
- [ ] Resolve open questions; Accept
- [ ] Workflow + `ci-gate` + concurrency
- [ ] Branch protection uses `ci-gate`
- [ ] Verify docs-only / code / Cython PRs
- [ ] Optional weekly full matrix

## References

- Workflow: `.github/workflows/test-and-deploy.yml`
- [CIP-0008](cip0008.md): classic main-branch release pipeline
- [CIP-0009](cip0009.md): coverage job (Ubuntu / 3.12)
- [CIP-0005](cip0005.md): ruff lint job
- Example of the cost: [#1247](https://github.com/SheffieldML/GPy/pull/1247)
  (backlog markdown only)
