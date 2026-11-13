---
author: Neil Lawrence
created: "2026-11-13"
id: "000F"
last_updated: "2026-11-13"
status: Proposed
compressed: false
related_requirements: []
related_cips:
- "0002"
- "0005"
- "0006"
- "000C"
- "000E"
tags:
- cip
- paramz
- parameterization
- architecture
- caching
- serialization
title: Staged reduction of the paramz dependency
---

# CIP-000F: Staged reduction of the paramz dependency

## Status

- [x] Proposed - Initial idea documented (open for Discussion feedback)
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

Community review: [Discussion #1293](https://github.com/SheffieldML/GPy/discussions/1293) (Ideas; same pattern as CIP-000E / [#1279](https://github.com/SheffieldML/GPy/discussions/1279)).

## Summary

Reduce GPy's coupling to [paramz](https://github.com/sods/paramz) in stages,
each of which is independently useful and leaves `pip install gpy` and the
public model API unchanged. The first stage is mechanical: route every paramz
import through `GPy.core.parameterization` and forbid direct imports elsewhere.
Later stages separate numerical code from `Param` objects, replace the
`Cache_this`/`ObsAr` observer-caching pair, and finally decide whether to vendor
the paramz core or replace it with a GPy-owned parameter tree.

CIP-0002 listed "replace paramz" as a very-high-effort option for parameter
tying. This CIP proposes a staged route that makes that option tractable, and
that also prepares the numerical core for a future autodiff backend (GPy2 /
JAX), without committing GPy 1.x to that backend.

## Motivation

- **paramz is a structural dependency, not a utility.** Today 63 non-test
  modules import paramz. `GPy.core.parameterization` already exists as a façade
  (it re-exports transformations and `ObsAr`), but ~55 modules bypass it — nearly
  every kernel, the psi-statistics, several models and three inference methods.
- **Recurring friction is already documented in this repo.**
  - CIP-0002: `tie_to` is documented in paramz but not implemented; tying is
    blocked on the parameter framework.
  - Backlog `2026-10-10_pickle-live-kernels`: root cause was a paramz pickle
    memento / `setattr` cycle, fixed GPy-side.
  - CIP-0005: Ruff `UP008` deferred because of the paramz metaclass.
  - `core/parameterization/parameterized.py` carries a paramz < 0.10.1
    compatibility workaround.
- **Hand-written gradients are coupled to the parameter framework.** Kernels,
  likelihoods and inference communicate through `.gradient` assignments (285
  sites), `parameters_changed` (39 overrides) and `Cache_this` (90 decorators).
  Separating pure numerical functions from parameter objects is valuable now
  (testability, clarity) and is the prerequisite for any autodiff backend later.
- **Maintenance surface.** paramz is effectively maintained alongside GPy;
  owning the narrow subset GPy actually uses is cheaper than coordinating
  releases of both.

## Detailed Description

### Current paramz surface (2026-11-13, `devel`)

| paramz symbol | GPy usage |
|---|---|
| `transformations.Logexp` (+ `Logistic`, `__fixed__`, `Transformation`) | 32+ modules |
| `caching.Cache_this` / `Cacher` | 21 + 5 modules, 90 decorators |
| `ObsAr` | 10 modules (models' X/Y, EP) |
| `Param`, `Parameterized`, `Model`, `Parameterizable` | core + `GPy.core.model` |
| `core.index_operations`, `lists_and_dicts`, `observable*`, `Pickleable` | core |
| `domains` (`_REAL`, `_POSITIVE`, `_NEGATIVE`) | priors, priorizable |
| `optimization` (SCG, L-BFGS-B wrappers) | `GPy.inference.optimization` |
| `link_parameter(s)` (method) | 135 call sites |
| constrain / fix / prior API (methods) | 69 call sites |

### Stages

Each stage is a separate PR series with its own acceptance check. Later stages
may be re-scoped at their own Accept point.

**Stage 0 — Single import boundary (no behaviour change).**
- Add `core/parameterization/caching.py` (re-export `Cache_this`, `Cacher`) and a
  `domains` re-export; rewrite all direct `paramz` imports outside
  `core/parameterization/` and `inference/optimization/` to GPy paths.
- Add a Ruff `TID251` (`banned-api`) rule forbidding `paramz` imports outside
  those packages.
- Exit: paramz imported in ≤ ~8 modules; full test suite unchanged.

**Stage 1 — Vendor loosely coupled pieces.**
- `domains` (constants) and `optimization` (SCG / L-BFGS-B wrappers; requires a
  GPy-owned `Model.optimize` dispatch).
- **Not** `transformations`: paramz core checks `isinstance(…, Transformation)`
  and compares `__fixed__` by identity, so these move only with the core
  (Stage 4).

**Stage 2 — Pure numerical functions.**
- Kernels: `K` / `Kdiag` (and, where present, psi-statistics forward passes)
  delegate to module-level functions of plain `ndarray`s; parameter values are
  unwrapped once at the method boundary.
- Likelihoods: same for `logpdf` and moments.
- Kernel by kernel, with current outputs as golden values. No gradient code is
  removed in GPy 1.x.

**Stage 3 — Replace `Cache_this` and `ObsAr` together.**
- They are coupled: `Cache_this` invalidates by observing `ObsAr`/`Param` inputs.
- Replace with an explicit version-counter cache (model bumps a counter in
  `parameters_changed` / `set_XY`; cache key includes it). `ObsAr` becomes plain
  arrays plus explicit `set_XY`.
- Benchmark before/after (e.g. `GPRegression`, `SparseGPRegression`,
  `BayesianGPLVM` optimisation wall-clock); remove caches that do not pay for
  themselves.

**Stage 4 — Core decision (separate Accept).**

| Option | Notes |
|---|---|
| A — Vendor paramz core into `GPy.core.parameterization` | Removes the external dependency quickly; keeps the design. |
| B — GPy-owned parameter tree (NumPy backend) | Designed with tying (CIP-0002), static/dynamic split and future pytree compatibility; more work. |
| C — Stop after Stage 3 | paramz confined to one package; acceptable steady state. |

### Non-goals

- Changing the user-facing model API (`m.kern.lengthscale = …`, `m.optimize()`,
  `print(m)`, `m['.*lengthscale']`).
- Removing hand-written gradients in GPy 1.x.
- Introducing JAX or any autodiff dependency into GPy 1.x.
- Cython changes (see Open questions).

### Alternatives considered

| Option | Verdict |
|---|---|
| Staged reduction (this CIP) | Preferred |
| Fix paramz upstream only (CIP-0002 Option 1) | Does not reduce coupling; tying still constrained by paramz design |
| Big-bang replacement | High risk; blocks other work; conflicts with incremental practice of CIP-0005/0009 |
| Do nothing | Friction above persists; GPy2 port has no clean numerical core to start from |

## Implementation Plan

1. **Proposed (this phase):** open Ideas Discussion; gather feedback on stage
   boundaries and Stage 4 options.
2. **Stage 0** after Accept: one PR; Ruff rule enabled in the same PR.
3. **Stage 1:** domains + optimisation; GPy-owned `optimize` dispatch.
4. **Stage 2:** kernel-by-kernel PRs (stationary family first), then
   likelihoods. Backlog task per family.
5. **Stage 3:** caching replacement behind benchmarks; record numbers in this
   CIP.
6. **Stage 4:** separate Accept decision with a short design note (may become
   its own CIP if Option B).

## Backward Compatibility

- **Public API:** unchanged through Stage 3. Stage 4 must preserve the
  documented constrain / fix / prior / indexing API.
- **Pickles:** saved models reference paramz class paths. Any stage that moves
  a pickled class (Stage 3 `ObsAr`; Stage 4 core) must ship a `sys.modules`
  alias or custom unpickler, and a test that loads pickles produced by the last
  release.
- **`to_dict` / `save` / `load`:** format unchanged (CIP-0006 paths kept green).
- **Third-party code importing `paramz` objects from GPy models** (e.g.
  `isinstance(p, paramz.Param)`): works through Stage 3; Stage 4 Option B needs
  a deprecation note.

## Testing Strategy

- Full `GPy/testing` suite unchanged and green at every stage.
- Golden-value tests for Stage 2: K, Kdiag, logpdf, log-likelihood and gradients
  of representative models recorded before refactoring.
- Pickle round-trip and "load previous-release pickle" fixtures before Stage 3.
- Wall-clock benchmarks for Stage 3 caching changes (record in this CIP).
- Ruff `TID251` in CI enforces the Stage 0 boundary.

## Related Requirements

None yet. Candidates after Accept:

- Parameter tying across kernel components (CIP-0002)
- Pickleable models and kernels (backlog `pickle-live-kernels`)
- Numerical core reusable by an autodiff backend

## Implementation Status

- [x] Draft CIP-000F (Proposed)
- [x] Ideas Discussion opened ([#1293](https://github.com/SheffieldML/GPy/discussions/1293))
- [ ] Accept: stage boundaries; pickle-compatibility policy
- [ ] Stage 0: single import boundary + Ruff rule
- [ ] Stage 1: domains, optimisation vendored
- [ ] Stage 2: pure numerical functions (kernels, likelihoods)
- [ ] Stage 3: caching / `ObsAr` replaced, benchmarks recorded
- [ ] Stage 4: core decision recorded
- [ ] Backlog tasks created (only after Accept)

## Open questions (resolve at Accept)

1. **CIP-0002:** fold tying into Stage 4 Option B, or keep CIP-0002 separate and
   make it depend on this CIP?
2. **Cython:** already [CIP-000E](cip000E.md)
   ([Discussion #1279](https://github.com/SheffieldML/GPy/discussions/1279)).
   This CIP does not decide it. Stages 0–3 do not depend on that outcome. If
   CIP-000E removes the Cython build hook, packaging gets simpler and the
   paramz stages stay the same.
3. **Caching policy:** keep any automatic caching, or only explicit
   model-level caches?
4. **Stage 4 default:** A (vendor) as an interim step before B, or go to B
   directly?

## References

- [CIP-0002](cip0002.md) — Parameter tying (Option 3: replace paramz)
- [CIP-0005](cip0005.md) — Ruff adoption (`UP008` deferred for paramz)
- [CIP-0006](cip0006.md) — Serialization maintenance
- [CIP-000C](cip000C.md) — uv packaging (Cython build-hook constraint)
- [CIP-000E](cip000E.md) — Cython hygiene and direction
- Backlog: `bugs/2026-10-10_pickle-live-kernels.md`,
  `infrastructure/2025-08-15_parameter-tying-framework.md`
- Tenet: [Community-Driven Development](../tenets/community-driven-development.md)
- paramz: https://github.com/sods/paramz
- Discussion: [#1293](https://github.com/SheffieldML/GPy/discussions/1293)
