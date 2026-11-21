---
id: 2026-11-21_cip000F-stage4-paramz-core-decision
title: CIP-000F Stage 4 — paramz core decision (vendor A interim, then B?)
status: Proposed
priority: Medium
created: "2026-11-21"
last_updated: "2026-11-21"
category: infrastructure
related_cips:
- "000F"
- "0002"
owner: Neil Lawrence
dependencies:
- 2026-11-21_cip000F-stage3-caching-obsar
tags:
- backlog
- cip000F
- paramz
---

# Task: CIP-000F Stage 4 paramz core decision

## Description

Separate Accept for Stage 4. Default from CIP-000F Accept (2026-11-21):
**Option A (vendor)** as interim, then decide whether to proceed to
**Option B** (GPy-owned parameter tree; enables CIP-0002 tying cleanly) or
stop at A/C. Record the decision; Option B may become its own CIP.

## Acceptance Criteria

- [ ] Short design note comparing A / B / C with current evidence
- [ ] Stage 4 Accept recorded in CIP-000F
- [ ] If A: vendored core; pickle aliases; external paramz dependency dropped
- [ ] If B follows: CIP-0002 dependency path clarified (or spun to new CIP)

## Related

- CIP: 000F, 0002

## Progress Updates

### 2026-11-21

Task created Proposed after CIP-000F Accept (default: A interim → decide on B).
