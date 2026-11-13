---
id: 2026-11-13_cip000D-deprecate-lfm1
title: CIP-000D — deprecate orphaned LFM1 kernel and schedule removal
status: Ready
priority: Medium
created: '2026-11-13'
last_updated: '2026-11-13'
category: infrastructure
related_cips:
- '000D'
- '0001'
owner: Neil Lawrence
dependencies:
- 2026-11-13_cip000D-kern-coverage-slices
tags:
- backlog
- deprecation
- kernels
- lfm
- coverage
- cip000D
---

# Task: Deprecate orphaned `LFM1` (`lfm1.py`)

## Description

CIP-000D K4 omitted `GPy/kern/src/lfm1.py` from coverage because it is
unexercised and superseded by `EQ_ODE*` / the CIP-0001 LFM direction. Coverage
omit alone is not enough for dead API: schedule an explicit deprecation and
removal.

**Contrast (do not deprecate under this task):** GPU `psi_comp/*gpucomp.py`
remains an optional live path (`useGPU` / SSGPLVM). Those files stay
coverage-omitted because CPU CI cannot run them — not because they are
scheduled for removal.

## Acceptance Criteria

- [ ] Confirm `LFM1` is not re-exported from `GPy.kern` (today: module-only)
- [ ] Add `DeprecationWarning` on construction / import of `LFM1`, pointing users
      at `EQ_ODE1` / CIP-0001
- [ ] Docstring / changelog note that the module will be removed after a stated
      release window (e.g. next minor after warning ships)
- [ ] Backlog or CIP-0001 note for the actual deletion once the window elapses
- [ ] Keep `.coveragerc` omit for `lfm1.py` until deletion (or drop omit when
      the file is gone)

## Implementation Notes

- Prefer warning at `LFM1.__init__` (and optionally module import) following
  CIP-0007-style deprecation patterns already used elsewhere in GPy.
- Do not remove the file in the same PR as the first warning unless Accept
  says otherwise — give consumers one release of signal.
- Coordinate wording with CIP-0001 so we do not promise a finished modern LFM
  API before that CIP lands.

## Related

- CIP: 000D (coverage triage), 0001 (LFM modernization)
- Depends on: 2026-11-13_cip000D-kern-coverage-slices
- Coverage omit: `.coveragerc` (`./GPy/kern/src/lfm1.py`)

## Progress Updates

### 2026-11-13

Task created **Ready** after CIP-000D discussion: omit ≠ deprecate; `LFM1`
needs a removal schedule, GPU psi does not.
