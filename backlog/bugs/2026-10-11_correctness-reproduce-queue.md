---
id: 2026-10-11_correctness-reproduce-queue
title: Reproduce historical correctness reports on current devel
status: Proposed
priority: Medium
created: '2026-10-11'
last_updated: '2026-10-11'
category: bugs
related_cips:
- '0006'
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- bugs
- reproduce
---

# Task: Correctness reproduce queue

## Description

Historical bug-shaped issues were consolidated into
[#1211](https://github.com/SheffieldML/GPy/issues/1211). Work the checklist on
current Poetry / NumPy 2 `devel`: fix, spawn focused tasks, or mark abandoned.

## Acceptance Criteria

- [ ] Each checklist item on #1211 reproduced, fixed, or abandoned with note
- [ ] New bugs get dedicated PRs/tests rather than silent checklist ticks
- [ ] Close #1211 when the queue is empty

## Related

- Issue: #1211
- CIP: 0006, 0007
