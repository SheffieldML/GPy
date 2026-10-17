---
id: 2026-10-11_deprecation-plotting-modernization
title: Deprecation warnings and plotting modernization
status: Completed
priority: Medium
created: '2026-10-11'
last_updated: '2026-10-17'
category: infrastructure
related_cips:
- '0005'
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- deprecations
- plotting
- pytest
---

# Task: Deprecation / plotting modernization

## Description

Tracking issue [#1210](https://github.com/SheffieldML/GPy/issues/1210): quiet or
fix deprecation noise (`#1072`, `#865`, `#968`) and retire `pylab` (`#913`).

## Acceptance Criteria

- [x] Capture top `pytest` warning categories on current `devel`
- [x] Fix or document intentional silences
- [x] Remove/replace `pylab` where practical
- [x] Close #1210 when done

## Inventory (2026-10-17, `pytest GPy/testing`)

On current Poetry / NumPy 2 `devel` the suite is already quiet versus the old
nose-era reports:

- No `MatplotlibDeprecationWarning` in a full run (legacy #865 issues gone).
- Remaining noise: traitlets HTML kwargs (third-party; filtered in pytest),
  intentional GPy `UserWarning`s about kernel input dim, and former SSGPLVM
  `init='rand'` deprecation (fixed as alias).

## Related

- Issue: #1210
- CIP: 0005, 0007
- PRs: #1229
- Follow-on: CIP-0009 (plotting / hotspot coverage)

## Progress Updates

### 2026-10-17
Completed via #1229; #1210 closed. Coverage follow-up tracked in CIP-0009.
