---
id: 2026-10-10_optimize-restarts-ep-state
title: Diagnose optimize_restarts worse than optimize under EP
status: Ready
priority: High
created: '2026-10-10'
last_updated: '2026-10-10'
category: bugs
related_cips:
- '0007'
owner: Neil Lawrence
contributor: olamarre (@olamarre)
dependencies: []
tags:
- backlog
- optimize
- expectation-propagation
- classification
---

# Task: `optimize_restarts` vs `optimize` under EP

## Description

[#1109](https://github.com/SheffieldML/GPy/issues/1109) (@olamarre):
`optimize_restarts` should be a superset of `optimize` (same start plus
random restarts) but can yield a *worse* fit. Martin Bubel traced that kernel
hyperparameters match while **ExpectationPropagation** variational /
site parameters differ between the two runs.

Active in 2025; not covered by the October 2026 correctness batch.

Open-issue triage (2026-10-10): high-priority backlog bug.

## Acceptance Criteria

- [ ] Minimal reproduction from #1109 on current `devel`
- [ ] Root cause documented (EP re-init, random state, site parameter copy)
- [ ] Fix so `optimize_restarts` is never worse than a single `optimize` from
      the same initial hyperparameters (same RNG seed policy documented)
- [ ] Regression test
- [ ] Comment / close #1109

## Implementation Notes

- Compare `to_dict()` of EP-related parameters across the two paths (Martin's
  approach in the issue thread).
- Check whether restarts reset EP state incorrectly or share mutable state.

## Related

- Issues: #1109
- Contributors: olamarre (@olamarre); investigation notes by Martin Bubel
- Open-issue triage: 2026-10-10
- Proposed parent CIP: 0007 (open-issue triage), when accepted

## Progress Updates

### 2026-10-10

Task created from open-issue triage.

- CIP-0007 High-backlog order: **next code fix after links** (Student-t #993
  already merged as #1160). Marked Ready.
