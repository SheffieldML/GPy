---
id: 2026-11-21_cip000F-stage3-caching-obsar
title: CIP-000F Stage 3 — replace Cache_this / ObsAr with version-counter cache
status: Proposed
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

- [ ] Wall-clock benchmarks recorded in CIP-000F before cache removal
- [ ] `ObsAr` move has `sys.modules` alias or custom unpickler + fixture
- [ ] Posterior-scoped state preferred over long-lived observer caches
- [ ] Tests green including pickle round-trip

## Related

- CIP: 000F

## Progress Updates

### 2026-11-21

Task created Proposed after CIP-000F Accept (starts after Stage 2 / 2N).
