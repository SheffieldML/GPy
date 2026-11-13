---
author: Neil Lawrence
created: '2026-11-13'
id: '000D'
last_updated: '2026-11-13'
status: Implemented
compressed: false

related_requirements: []
related_cips:
- '0001'
- '0009'
tags:
- cip
- testing
- coverage
- kernels
- kern
title: Kernel package coverage (GPy.kern) follow-on to CIP-0009
---

# CIP-000D: Kernel package coverage (GPy.kern)

## Status

- [x] Proposed - Initial idea documented
- [x] Accepted - Approved, ready to start work
- [x] In Progress - Actively being implemented
- [x] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

CIP-0009 raised **patch hygiene** and util/plotting smokes; project coverage is
still ~**57%**. The largest remaining pure-Python hole with clear APIs is
**`GPy/kern`** (~**66%** of statements covered; **~2.3k** missing of ~6.9k).
This CIP plans **meaningful** kernel coverage gains by slicing the miss mass,
not chasing 90%+ package coverage.

**Baseline (2026-11-13, full `pytest GPy/testing` under `coverage.py`):**

| Scope | Cover |
|-------|-------|
| Project (`GPy/`) | ~57% |
| `GPy/kern/` | ~66% (2321 miss / 6856 stmts) |

## Motivation

### What is going on in `kern`

Gradient / PSD checks in `GPy/testing/test_kernel.py` already exercise the
**core stationary / compose path** well (`RBF`, `Matern*`, `Add`, `Prod`,
`Linear`, `MLP`, `StdPeriodic`, …). Coverage concentrates there. The miss mass
is **not** “kernels are untested” in general — it is a few large, cold modules
plus optional / legacy paths.

**Miss by category (approx., statements):**

| Category | Miss | Cover (files) | Notes |
|----------|------|---------------|-------|
| `eq_ode1.py` + `eq_ode2.py` | ~986 | 44% / 20% | Largest hole. `test_lfm_kernel.py` exists (CIP-0001 related) but only hits construction / subset of `K` paths; huge analytic blocks unused |
| Legacy `ODE_*.py` (`ODE_UYC`, `ODE_st`, `ODE_t`, part of `ODE_UY`) | ~384 | mostly ~5% | Only `ODE_UY` appears in `test_kernel.py`; siblings almost never constructed |
| Legacy `periodic.py` (`PeriodicExponential` / `Matern32` / `Matern52`) | ~148 | ~48% | **Zero** mentions in `test_kernel.py`; `StdPeriodic` (modern) is ~98% |
| GPU `psi_comp/*_gpucomp.py` | ~230 | ~11–12% | Optional GPU path; CI has no GPU — expect low cover unless mocked / excluded |
| `lfm1.py` | ~94 | **0%** | Dead / superseded by EQ_ODE* + CIP-0001 direction |
| Sparse `ssrbf_psi_comp.py` | ~59 | ~5% | Sparse variational ψ stats for RBF; thin vs dense `rbf_psi_comp` (100%) |
| Core stationary / slice / prod / add residual | ~170 | already 82–91% | Diminishing returns |
| Small misc (`Spline`, integrals diag, `TruncLinear` edges, …) | ~195 | mixed | Cheap wins if we want package % |

### Why this matters

- Kernel PRs that touch `eq_ode*` / legacy `periodic` / ODE helpers will keep
  showing **angry Codecov patch** reports (same failure mode as CIP-0009’s
  plotting modernization).
- CIP-0001 (LFM modernization) and CIP-0009 (coverage process) need a
  **follow-on HOW** that is kernel-specific and prioritised by miss mass.
- GPU and dead modules should be **explicitly classified** (test, mock, or
  omit from coverage) so reported % means something.

## Detailed Description

### Goals

1. **Triage map** — keep this CIP’s category table updated; mark each file as
   *cover*, *mock*, *omit* (`.coveragerc` / pragma), or *defer to CIP-0001*.
2. **Slice K1 — modern LFM / EQ_ODE path:** deepen `test_lfm_kernel.py` so
   `EQ_ODE1` / `EQ_ODE2` hit `K`, `Kdiag`, and gradient checks for
   representative index layouts (outputs + latent). Target: cut miss in
   `eq_ode*.py` substantially without rewriting the analytic code.
3. **Slice K2 — legacy periodic API:** add `check_kernel_gradient_functions`
   (or equivalent) for `PeriodicExponential`, `PeriodicMatern32`,
   `PeriodicMatern52` *or* document deprecation in favour of `StdPeriodic`
   and omit cold branches with rationale.
4. **Slice K3 — legacy ODE kernels:** smoke + gradient for `ODE_UYC`,
   `ODE_st`, `ODE_t` where cheap; otherwise mark deferred / omit if unmaintained.
5. **Slice K4 — policy for GPU / dead code:**
   - GPU psi: omit from coverage upload (or `pytest.importorskip` + mock) —
     do not inflate expectations on CPU CI.
   - `lfm1.py`: align with CIP-0001 (cover only if kept; else omit / remove
     in a later CIP).
6. **Slice K5 (optional):** cheap misc — `Spline`, remaining integral
   diagonals, `ssrbf_psi_comp` on CPU if sparse models already in suite.

### Non-goals

- 90%+ `GPy/kern` or project coverage.
- Rewriting EQ_ODE analytics or finishing CIP-0001’s new LFM API in this CIP.
- Making Codecov patch a hard merge gate (stays soft / informational per CIP-0009).
- Full GPU testing in GHA.

### Relation to other CIPs

- **CIP-0009** — process + plotting/util; this CIP is the **kernel** follow-on.
- **CIP-0001** — LFM product direction; K1 deepens tests on *current* EQ_ODE
  surfaces; new unified LFM kernel remains CIP-0001’s job.

## Implementation Plan

1. Freeze baseline numbers in this CIP (done above); re-run
   `coverage run -m pytest GPy/testing` after each slice.
2. **K1:** extend `test_lfm_kernel.py` / gradient helpers for EQ_ODE1/2
   (prioritise `eq_ode2.py` miss).
3. **K2:** decide cover vs deprecate for `GPy.kern.Periodic*`; implement.
4. **K3:** same for `ODE_*` siblings.
5. **K4:** update `.coveragerc` omit list for GPU psi (and `lfm1` if dead)
   with a one-line comment pointing at this CIP.
6. **K5 (optional):** small kernels with clear APIs.
7. Close when K1–K4 done and kern cover / patch hygiene documented in README
   (link from CIP-0009 testing note).

## Backward Compatibility

Tests and coverage config only, unless K2/K3 choose documented deprecation
(then follow CIP-0007-style warnings, no silent removal).

## Testing Strategy

- Prefer existing `check_kernel_gradient_functions` / PSD helpers in
  `test_kernel.py`.
- EQ_ODE tests must set index columns correctly (outputs vs latent) as in
  `test_lfm_kernel.py` setup.
- GPU modules: never fail CPU CI; omit or skip.
- Success metric: `GPy/kern` statement cover clearly above 66% **or** miss mass
  in eq_ode/periodic/ODE reduced with remaining miss explained by omit list;
  kernel PRs that touch covered APIs show healthy Codecov patch %.

## Related Requirements

None yet.

## Implementation Status

- [x] Baseline inventory (2026-11-13)
- [x] Accept CIP after review (2026-11-13)
- [x] K1: EQ_ODE1 / EQ_ODE2 deep tests (`TestEQODECoverageCIP000D`)
- [x] K2: legacy Periodic* gradient coverage
- [x] K3: legacy ODE_UYC / ODE_st / ODE_t gradient smokes
- [x] K4: GPU psi / `lfm1` omitted in `.coveragerc`
- [x] K5: Spline gradient smoke
- [ ] Close CIP (after merge / validation)

**Post-slice baseline (2026-11-13):** `GPy/kern` ~**80%** (was ~66%);
`periodic.py` ~99%; `ODE_*` ~87–88%; `eq_ode1` ~65%; `eq_ode2` ~38%
(remaining miss is largely unimplemented `Kff` full and deep analytic
branches — deferred with CIP-0001).

**Follow-on backlog:** deprecate orphaned `LFM1` and schedule removal
(`backlog/infrastructure/2026-11-13_cip000D-deprecate-lfm1.md`). GPU
`*gpucomp.py` stays omit-only (optional path; not a deprecation).

## References

- CIP-0009 — incremental coverage / soft Codecov patch 50%
- CIP-0001 — LFM kernel modernization (Proposed)
- `GPy/testing/test_kernel.py` — gradient / compose suite
- `GPy/testing/test_lfm_kernel.py` — EQ_ODE1/2 starter suite
- Codecov / local: `coverage run -m pytest GPy/testing` then
  `coverage report --include='GPy/kern/*'`
