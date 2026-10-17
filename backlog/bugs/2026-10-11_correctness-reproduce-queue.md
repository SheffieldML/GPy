---
id: 2026-10-11_correctness-reproduce-queue
title: Reproduce historical correctness reports on current devel
status: In Progress
priority: Medium
created: '2026-10-11'
last_updated: '2026-10-17'
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

## Progress Updates

### 2026-10-17
Moved to In Progress. Working #1211 checklist on current `devel`.

Slice 1 (PR #1222): #639 Uniform `lnpdf`, #871 X/Y mismatch, #752 GPLVM `plot_inducing`.
Smoke cleared: #728, #738, #510, #641 (simple), #919, #734.

Slice 2 (PR #1223): #1039 Fixed kernel predict shape error, #839 GPClassification
multi-column Y, #828 ExpQuad/RBF docstring lengthscale-scaled ``r``.
