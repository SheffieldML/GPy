---
id: 2026-11-21_cip000F-stage4-paramz-core-decision
title: CIP-000F Stage 4 — paramz core decision (Accept A vendor; 1.20 line)
status: Completed
priority: Medium
created: "2026-11-21"
last_updated: "2026-11-22"
category: infrastructure
related_cips:
- "000F"
- "0002"
- "0010"
owner: Neil Lawrence
dependencies:
- 2026-11-21_cip000F-stage3-caching-obsar
tags:
- backlog
- cip000F
- paramz
- release-1.20
---

# Task: CIP-000F Stage 4 paramz core decision

## Description

Separate Accept for Stage 4, targeted at **GPy 1.20** (not 1.16). Through
1.16, paramz remains a declared dependency; only tidy / confinement work
ships (CIP-000F Stages 0–3 and follow-ons).

Default lean from CIP-000F Accept / 2026-11-22 addendum:

- **Option A (vendor)** remains an allowed *interim* if needed at 1.20.
- Prefer progressing to **Option B** (GPy-owned parameter tree) with
  **B2 lean confirmed** (2026-11-22):
  - **B1** — adopt an explicit pytree discipline early (only on evidence)
  - **B2** — pytree-compatible NumPy representation first (**lean**)
- B2 / the 1.16↔1.20 split is **input to CIP-0010's GPy-1 compatibility
  pass**; that pass may send tidy improvements into 1.16 without pulling
  dependency removal forward of 1.20.
- Option B may become its own CIP if large; clarifies CIP-0002 tying path.

## Acceptance Criteria

- [x] Short design note comparing A / B / C **and** B1 vs B2 lean, with
      Stage 2T / CIP-0010 evidence cited
- [x] Stage 4 Accept recorded in CIP-000F — **Accept A (vendor)**; B deferred
- [x] Implementation (Accept A): vendored core + pickle aliases + external
      paramz dropped (dependency-removal vehicle on the **1.20** line)
- [x] If B follows: CIP-0002 dependency path clarified (or spun to new CIP)
      — B deferred; CIP-0002 remains separate
- [x] Release mapping recorded: tidy → **1.16**; dependency removal → **1.20**

## Related

- CIP: 000F, 0002, 0010
- Notes: [`cip/cip000F/notes/stage4-core-decision.md`](../../cip/cip000F/notes/stage4-core-decision.md)
- Release: dependency removal → **1.20**; tidy → **1.16**

## Progress Updates

### 2026-11-21

Task created Proposed after CIP-000F Accept (default: A interim → decide on B).

### 2026-11-22

Release split recorded in CIP-000F: tidy in 1.16, dependency removal in 1.20.
**B2 lean confirmed.** Cross-link to CIP-0010 Workstream E: B2 and the split
are compatibility-pass inputs; findings may improve 1.16 without advancing
paramz removal.

### 2026-11-22 (Stage 4 Accept A)

Accept A recorded and implemented: `_paramz` vendor tree, bootstrap aliases,
`paramz` removed from `pyproject.toml`, `decorator` added. B remains deferred.
