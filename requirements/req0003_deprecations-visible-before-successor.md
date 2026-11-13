---
id: "0003"
title: "Deprecations are visible at least one GPy1 minor release before the successor's first stable release"
status: "Proposed"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets:
- "community-driven-development"
stakeholders:
- "GPy users of features that will not carry forward"
- "GPy maintainers"
tags:
- gpy2
- deprecation
- communication
---

# REQ-0003: Deprecations are visible at least one GPy1 minor release before the successor's first stable release

## Description

Users who rely on features that will not carry into the successor learn
about that under GPy1, through warnings that are visible by default and
through a maintained deprecation table — early enough to plan, and at
least one GPy1 minor release before the successor's first stable release.

Working user-facing features are not removed during GPy 1.x solely
because they are successor-bound; dead code that cannot run may still be
removed with a CHANGELOG entry.

**Why this matters**: Surprising removals at a major upgrade violate
community ownership of the project. Visible, documented deprecation is
how consensus and evidence from Discussion become operational.

**Who benefits**: Users of MPI, GPU, MCMC, datasets, or other candidates;
contributors who need a single policy instead of ad hoc warnings.

## Acceptance Criteria

- [ ] GPy1 no longer silences all `DeprecationWarning`s process-wide on
      import (narrow third-party filters only where justified)
- [ ] User-facing successor-bound deprecations emit `FutureWarning` with a
      named replacement or explicit "not carried forward" pointer
- [ ] A `DEPRECATIONS.md` (or equivalent) lists each deprecation with
      version deprecated, replacement, and successor status
- [ ] Every feature Discussion confirms will not carry forward has shipped
      as a visible deprecation for at least one GPy1 minor release before
      the successor's first stable release
- [ ] Tests assert the warning once per deprecated entry point

## Notes

- The exact minimum window (one vs N minor releases) is CIP-0010 open
  question 2; "at least one" is the floor this requirement states.
- Which candidates are deprecated is decided with Discussion #1297, not
  in this requirement.
- Dead-code removal (e.g. `scipy.weave`) is allowed without a long
  deprecation window because the code cannot run on supported SciPy.

## References

- Related tenet: [Community-Driven Development](../tenets/community-driven-development.md)
- CIP-0010: [GPy2 readiness](../cip/cip0010.md)
- Discussion: https://github.com/SheffieldML/GPy/discussions/1297

## Progress Updates

### 2026-11-13

Requirement drafted from CIP-0010 candidate list; status Proposed pending
Discussion #1297 and CIP Accept.
