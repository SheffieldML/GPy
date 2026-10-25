---
id: 2026-10-11_example-datasets-links
title: Fix or soft-fail example dataset downloads
status: Superseded
priority: Low
created: '2026-10-11'
last_updated: '2026-10-25'
category: documentation
related_cips:
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- datasets
- examples
---

# Task: Example datasets / broken links

## Description

Tracking [#1214](https://github.com/SheffieldML/GPy/issues/1214): `#841` missing
example datasets, `#671` Olympic Sprints link.

## Acceptance Criteria

- [ ] Links fixed, mirrored, or examples skip cleanly when data absent
- [ ] Close #1214 when done

## Related

- Issue: #1214
- CIP: 0007
- Superseded by: `2026-10-25_sync-dataset-resource-urls`

## Progress Updates

### 2026-10-11

Task created Proposed as the CIP-0007 tracker stub for #1214.

### 2026-10-25

Superseded. The remaining work is a JSON URL sync against pods 0.1.19 /
`datasets_mirror`, captured in `2026-10-25_sync-dataset-resource-urls`.
