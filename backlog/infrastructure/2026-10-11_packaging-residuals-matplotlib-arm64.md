---
id: 2026-10-11_packaging-residuals-matplotlib-arm64
title: Packaging residuals — matplotlib import coupling and arm64 Cython wheels
status: Proposed
priority: High
created: '2026-10-11'
last_updated: '2026-10-11'
category: infrastructure
related_cips:
- '0003'
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- packaging
- matplotlib
- wheels
- arm64
---

# Task: Packaging residuals (matplotlib + arm64)

## Description

After CIP-0003, two packaging gaps remain (GitHub tracking #1209):

1. Importing `GPy` pulls in plotting and thus requires matplotlib even for
   non-plotting use (`#873`, `#980`, `#1034`).
2. Arm64 / some platforms report missing Cython extensions (`#1114`, `#902`).

## Acceptance Criteria

- [ ] Decide lazy-import vs hard matplotlib dependency; implement + document
- [ ] Document and, where feasible, ship arm64 / aarch64 wheels aligned with CI
- [ ] Close #1209 when done

## Related

- Issue: [#1209](https://github.com/SheffieldML/GPy/issues/1209)
- CIP: 0003, 0007
