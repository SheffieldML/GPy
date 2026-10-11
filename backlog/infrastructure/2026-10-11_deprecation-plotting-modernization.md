---
id: 2026-10-11_deprecation-plotting-modernization
title: Deprecation warnings and plotting modernization
status: Proposed
priority: Medium
created: '2026-10-11'
last_updated: '2026-10-11'
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

- [ ] Capture top `pytest` warning categories on current `devel`
- [ ] Fix or document intentional silences
- [ ] Remove/replace `pylab` where practical
- [ ] Close #1210 when done

## Related

- Issue: #1210
- CIP: 0005, 0007
