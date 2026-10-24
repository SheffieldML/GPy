---
id: 2026-10-24_cip0008-rename-default-main
title: CIP-0008 — rename default branch devel to main
status: In Progress
priority: High
created: "2026-10-24"
last_updated: "2026-10-24"
category: infrastructure
related_cips:
- "0008"
owner: Neil Lawrence
dependencies:
- 2026-10-24_cip0008-ci-main-release
tags:
- backlog
- infrastructure
- git
- dependabot
---

# Task: Rename default branch `devel` → `main`

## Description

After the CIP-0008 CI change is on `devel`, rename the GitHub default branch
`devel` → `main` (no `main` exists today). GitHub retargets open PRs and
redirects old URLs.

Then: Dependabot `target-branch: main`; drop `devel` from workflow `push:`
if it is still listed; README badges and RTD `en/devel` links; archive or
delete the stale `deploy` branch (last commit July 2024).

## Acceptance Criteria

- [x] Repository default branch is `main`
- [x] Dependabot PRs open against `main`
- [x] CI `push` triggers are `main` only (no `deploy`, no leftover `devel`
      unless a redirect branch remains)
- [x] README contributor/release instructions say `main`
- [ ] `deploy` removed from CI; branch deleted or clearly archived

## Related

- CIP: 0008
- Depends on: 2026-10-24_cip0008-ci-main-release

## Progress Updates

### 2026-10-24

Task created Ready. Do this after the CI PR merges so `devel` still receives
the workflow that knows about `main`.

GitHub default branch renamed `devel` → `main`. Follow-up PR retargets
Dependabot, drops leftover `devel` from `push:`, and updates README/RTD
links. The stale `deploy` branch is left in place (not a CI trigger).
