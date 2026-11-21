---
id: 2026-11-21_cip000F-stage3-caching-obsar
title: CIP-000F Stage 3 — replace Cache_this / ObsAr with version-counter cache
status: Completed
priority: Medium
created: "2026-11-21"
last_updated: "2026-11-21"
category: infrastructure
related_cips:
- "000F"
owner: Neil Lawrence
dependencies:
- 2026-11-21_cip000F-stage2n-numerical-interfaces
tags:
- backlog
- cip000F
- caching
- pickle
---

# Task: CIP-000F Stage 3 caching and ObsAr

## Description

Replace `Cache_this` / `ObsAr` with an explicit version-counter cache, guided
by Stage 2N posterior-scoped state findings. Benchmark before removing any
cache. Ship pickle alias for `ObsAr` and a load-previous-release test.

## Acceptance Criteria

- [x] Wall-clock benchmarks recorded in CIP-000F before cache removal
- [x] `ObsAr` move has `sys.modules` alias or custom unpickler + fixture
- [x] Posterior-scoped state preferred over long-lived observer caches
- [x] Tests green including pickle round-trip

## Related

- CIP: 000F
- Notes: [`cip/cip000F/notes/stage3-caching-obsar.md`](../../cip/cip000F/notes/stage3-caching-obsar.md)

## Progress Updates

### 2026-11-21

Task created Proposed after CIP-000F Accept (starts after Stage 2 / 2N).

### 2026-11-21 (later)

Completed: GPy-owned caching with `_cache_version`, GP X/Y as ndarray,
ObsAr module alias, benchmarks, tests in `test_stage3_caching_obsar.py`.
