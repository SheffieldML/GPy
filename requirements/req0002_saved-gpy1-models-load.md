---
id: "0002"
title: "Saved GPy1 models load into the successor"
status: "Proposed"
priority: "High"
created: "2026-11-13"
last_updated: "2026-11-13"
related_tenets:
- "community-driven-development"
stakeholders:
- "GPy users with saved models"
- "GPy maintainers"
tags:
- gpy2
- serialization
- migration
---

# REQ-0002: Saved GPy1 models load into the successor

## Description

A user who has a model fitted and saved under GPy1 can load an equivalent
model under the successor, recovering parameters and being able to
predict, without needing the successor to unpickle GPy1 objects.

The supported migration path is export under GPy1 to a versioned
JSON-compatible representation (`to_dict` / documented schema), then load
via a successor entry point (e.g. `from_gpy1`). Pickle is not required to
work across the boundary.

**Why this matters**: Saved models are research artefacts and production
state. Losing them at an upgrade splits the community as surely as
breaking scripts.

**Who benefits**: Users with long-lived fitted models; anyone converting
legacy pickles under GPy1 before moving.

## Acceptance Criteria

- [ ] GPy1 can export every first-pass golden-corpus model via `to_dict`
      (and round-trip `from_dict` under GPy1)
- [ ] The JSON schema is versioned and documented
- [ ] A documented GPy1 procedure converts existing pickles to that format
- [ ] The successor can load first-pass corpus exports and produce
      predictions that match the golden corpus within stated tolerances
- [ ] Inventory models without serialisation are explicitly marked
      unsupported (no silent gaps)

## Notes

- CIP-0010's readiness bar is GPy1 export + schema for the first-pass
  set; successor load is verified when the GPy2 CIP implements
  `from_gpy1` (or equivalent).
- Expanding `to_dict` to the full public inventory may trail the
  readiness bar as backlog.

## References

- Related tenet: [Community-Driven Development](../tenets/community-driven-development.md)
- CIP-0010: [GPy2 readiness](../cip/cip0010.md)
- Related: CIP-0006 (prior serialisation maintenance)
- Discussion: https://github.com/SheffieldML/GPy/discussions/1297

## Progress Updates

### 2026-11-13

Requirement drafted from CIP-0010 candidate list; status Proposed pending
Discussion #1297 and CIP Accept.
