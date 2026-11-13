---
id: "0001"
title: "GPy1 scripts run under the successor compatibility layer with at most an import change"
status: "Proposed"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets:
- "community-driven-development"
stakeholders:
- "GPy users with existing scripts and tutorials"
- "GPy maintainers"
tags:
- gpy2
- compatibility
- migration
---

# REQ-0001: GPy1 scripts run under the successor compatibility layer with at most an import change

## Description

A user who has a working GPy1 script that only uses the recorded public API
can run that script against the successor's compatibility layer by changing
the import (and package install) — without rewriting model construction,
optimisation, prediction, constraint, or indexing calls.

"Public API" means the surface captured by CIP-0010's inventory (documented /
`__all__` / behavioural checklist), not every private helper in the GPy1
tree.

**Why this matters**: Upgrade ease is what keeps the community together.
Community-driven development means the project does not force a rewrite on
users who followed the documented API.

**Who benefits**: Anyone with scripts, notebooks, or published tutorials
built on GPy1; maintainers who need a falsifiable upgrade claim.

## Acceptance Criteria

- [ ] A checked-in public API inventory exists and is diffed in CI
- [ ] The successor ships a compatibility layer that covers that inventory
      (or documents explicit gaps before first stable release)
- [ ] For each first-pass golden-corpus model, a GPy1-style script runs
      under the compatibility layer with only import / install changes
- [ ] Behavioural checks pass for `optimize`, `predict` (and related),
      constraints, regex indexing, and `print(m)` where listed in the
      inventory

## Notes

- Side-by-side install under a separate import name is the working
  assumption of CIP-0010; the exact name is an Accept / Discussion
  question.
- Custom kernels and likelihoods are covered only insofar as they follow
  the forward-compatible helper (CIP-0010 open question 6); arbitrary
  subclasses that reach into private paramz APIs are out of scope for
  this requirement.
- Implemented primarily by CIP-0010 (GPy1 preparation) and the future
  GPy2 design CIP (compatibility layer).

## References

- Related tenet: [Community-Driven Development](../tenets/community-driven-development.md)
- CIP-0010: [GPy2 readiness](../cip/cip0010.md)
- Discussion: https://github.com/SheffieldML/GPy/discussions/1297

## Progress Updates

### 2026-11-13

Requirement drafted from CIP-0010 candidate list; status Proposed pending
Discussion #1297 and CIP Accept.
