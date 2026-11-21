---
id: 2026-11-21_cip000F-stage1-domains-optimisation
title: CIP-000F Stage 1 — vendor domains and optimisation wrappers
status: Ready
priority: Medium
created: "2026-11-21"
last_updated: "2026-11-21"
category: infrastructure
related_cips:
- "000F"
owner: Neil Lawrence
dependencies:
- 2026-11-21_cip000F-stage0-import-boundary
tags:
- backlog
- cip000F
- paramz
- optimisation
---

# Task: CIP-000F Stage 1 domains and optimisation

## Description

Vendor `domains` constants and SCG / L-BFGS-B optimisation wrappers into GPy,
with a GPy-owned `Model.optimize` dispatch. Leave `transformations` in paramz
(identity checks on `Transformation` / `__fixed__`).

## Acceptance Criteria

- [ ] `domains` no longer imported from paramz outside the façade
- [ ] Optimisation wrappers owned by GPy; `Model.optimize` dispatch works
- [ ] Transformations still come from paramz
- [ ] Tests green; pickle / `to_dict` paths unchanged

## Related

- CIP: 000F

## Progress Updates

### 2026-11-21

Task created Ready after CIP-000F Accept.
