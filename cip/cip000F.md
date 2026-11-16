---
author: Neil Lawrence
created: "2026-11-13"
id: "000F"
last_updated: "2026-11-16"
status: Proposed
compressed: false
related_requirements: []
related_cips:
- "0002"
- "0005"
- "0006"
- "000C"
- "000E"
- "0010"
tags:
- cip
- paramz
- parameterization
- architecture
- caching
- serialization
- transformation-readiness
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
tying. This CIP proposes a staged route that makes that option tractable.

**Central principle for Stage 2 / 2T / 2N:** GPy 1 remains a NumPy/SciPy
library, but its numerical core is progressively refactored to support
functional program transformations and to expose the smallest useful
covariance / posterior / parameter protocols. JAX is used as a
**diagnostic and experimental target** — a test of GPy's architectural
assumptions — not adopted prematurely as the successor's backend. The aim
is not to imitate GPyTorch, GPflow or GPJax, but to discover which
abstractions existing GPy algorithms need. That improves GPy 1
independently of whether GPy 2 ultimately adopts JAX; backend selection
remains open and is informed by evidence under [CIP-0010](cip0010.md).

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

### Current paramz surface (2026-11-13, `main`)

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

**Design principle (Stage 2) — Transformation readiness where practical.**

Numerical functions extracted from GPy's parameterisation framework should be
designed, where practical, to support functional program transformations.

These functions must continue to operate using NumPy/SciPy in GPy 1. They
should not introduce a runtime dependency on JAX or alter the existing public
API.

Transformation readiness means:

- Explicit numerical inputs, parameters and outputs.
- No observable mutation of model state within numerical calculations.
- No implicit dependence on parameter observers, global random state or
  hidden caches.
- A clear distinction between static model structure and dynamic numerical
  parameter values.
- Explicit treatment of randomness where applicable.
- Avoidance of unnecessary value-dependent Python control flow and
  dynamically sized intermediates in selected numerical paths.
- Explicit identification of numerical operations requiring backend-specific
  implementations or derivative rules.

This is a design direction, not an absolute restriction. Existing NumPy
implementations should not be made less efficient or less readable merely to
satisfy JAX constraints. Every production change must still have an
independent justification in correctness, testability, maintainability,
performance or clarity (see [Shared implementation guidance](#shared-implementation-guidance)
below).

**Stage 2 acceptance criteria (production GPy 1):**

- At least one representative GP objective is implemented as a pure
  NumPy/SciPy numerical function.
- Existing GPy numerical outputs and gradients remain within documented
  tolerances.
- The extracted function can be evaluated without creating a GPy model or
  `paramz` parameter object.
- GPy 1 retains its existing public API and NumPy/SciPy runtime.

**Non-goal:** Making the entire GPy 1 codebase JAX-compatible.

**Stage 2T — Transformation-readiness experiments (extends Stage 2; does not
compete with it).**

Stage 2 already extracts pure numerical functions. Stage 2T does **not**
start a parallel refactoring programme: it adds explicit
transformation-readiness objectives and out-of-package experiments to the
same Stage 2 extractions.

After extracting representative pure numerical functions, implement
experimental JAX equivalents *outside* the GPy 1 runtime dependency graph.

**Important distinction:** running a function on JAX arrays ("JAX-compatible
NumPy code") is not the same as an algorithm that transforms efficiently
under differentiation, JIT or vectorisation ("JAX-compatible algorithms").
Some GP procedures will need genuinely different formulations to become
efficiently compilable or differentiable. That is precisely what the
experiments should reveal — not something Stage 2 is expected to solve
up front.

Use these experiments to test:

1. Numerical agreement with GPy 1 reference values.
2. Agreement between automatic and existing analytical gradients.
3. Compatibility with `jax.jit`.
4. Compatibility with `jax.vmap` where batching is meaningful.
5. Behaviour under explicit floating-point precision settings.
6. Differences in runtime, memory requirements and numerical stability.

A function successfully evaluated using JAX arrays should not automatically
be described as transformation-compatible. Compatibility must be demonstrated
under the relevant transformations. Where an algorithm fails under `jit` /
`grad` / `vmap`, classify the barrier (below) and record whether a different
*formulation* (not merely a different array library) would be required.

**Initial test cases** (incremental; not a prerequisite for completing all of
Stage 2):

- RBF kernel and exact GP log marginal likelihood.
- Exact GP predictions and their derivatives.
- Sparse variational GP inference with inducing inputs.
- RBF psi-statistics and their derivatives.
- Bayesian GPLVM objective and gradients.

**Suggested first experiment:** extract the exact GP regression log marginal
likelihood into a pure NumPy/SciPy function, retaining the existing GPy
interface as a wrapper; verify likelihood and gradient parity against existing
fixtures; implement a separate JAX equivalent and test numerical equivalence,
`jax.grad`, `jax.jit` and handling of static model structure. Prefer returning
an explicit **conditioned numerical state** (factorisation / solver artefacts)
that a second pure function uses for repeated prediction, so the experiment
also probes the `ConditionedState` idea below. Completion means producing
useful evidence, not achieving perfect JAX compatibility.

**Stage 2N — Numerical architecture investigation (conceptual; light-touch).**

GPyTorch, GPflow and GPJax made different architectural choices; GPy 2 should
learn from all three and imitate none wholesale. Alongside Stage 2 / 2T, the
pure-function extraction should investigate three *conceptual* interfaces —
not mandatory class names, and **not** a large hierarchy introduced into
production GPy 1:

| Interface (conceptual) | Purpose | Inspired by |
|---|---|---|
| `CovarianceOperator` | Kernel / diagonal evaluation; apply covariance to a vector or matrix; structured solves; log-determinant (exact or approximate) without assuming every \(K\) is a dense NumPy array | GPyTorch |
| `ConditionedState` | Explicit factorisations and reusable posterior calculations for repeated prediction | GPJax |
| `ParameterSpec` | Distinguish static model structure, free numerical parameters, transformations and constraints | GPJax, GPflow |

Distinguish mathematical operations early, even when the only production
implementation remains dense Cholesky:

- Evaluate a kernel or kernel diagonal.
- Apply a covariance operator to a vector or matrix.
- Solve a linear system.
- Compute or approximate a log determinant.
- Construct a Gaussian posterior using the resulting factorisation or solver
  state.

**Caveat:** do **not** immediately introduce GPyTorch-scale `LinearOperator`
class hierarchies into GPy 1 — that would create enormous churn. Identify the
smallest useful numerical protocol and exercise it in experimental
implementations. Traditional dense Cholesky paths remain supported.
Flexibility in mathematical operations matters more than copying any one
library's implementation architecture.

**Acceptance for Stage 2N:** a minimal working experiment (e.g. exact GP
conditions once and predicts repeatedly via pure numerical state) with
outputs unchanged from GPy 1 — not a comprehensive abstraction layer.

**Record transformation barriers.** Each experiment should document any
encountered limitation and classify it as one of:

| Class | Meaning |
|---|---|
| **Incidental implementation coupling** | Mutable state, implicit caching, parameter-object dependencies or hidden side effects. |
| **Control-flow or shape limitation** | Operations requiring changes for tracing or compilation. |
| **Numerical limitation** | Conditioning, precision, factorisation or differentiation behaviour. |
| **Backend-specific optimisation** | An operation better implemented differently for NumPy and JAX. |
| **Fundamental modelling constraint** | A difficulty intrinsic to the algorithm rather than its current implementation. |

For each barrier, record whether it can be addressed in GPy 1 without
compromising existing behaviour. Findings feed
[CIP-0010](cip0010.md) Workstream G (transformation-readiness report).

**Stage 2T acceptance criteria (experiments):**

- A separate experimental JAX implementation has been assessed for
  differentiation and JIT compatibility for at least one representative
  objective.
- Findings, including unsuccessful transformations, are documented.
- Experimental code does not become a GPy package runtime dependency.

**Stage 3 — Replace `Cache_this` and `ObsAr` together.**
- They are coupled: `Cache_this` invalidates by observing `ObsAr`/`Param` inputs.
- Replace with an explicit version-counter cache (model bumps a counter in
  `parameters_changed` / `set_XY`; cache key includes it). `ObsAr` becomes plain
  arrays plus explicit `set_XY`.
- Benchmark before/after (e.g. `GPRegression`, `SparseGPRegression`,
  `BayesianGPLVM` optimisation wall-clock); remove caches that do not pay for
  themselves.

**Caching reframed (lesson from GPJax):** functional numerical cores do not
mean "no caching". GPJax 1.0's `condition` returns a posterior that caches the
factorisation needed for repeated predictions — but that cache belongs to an
*explicit* posterior for specific data and parameters, not to mutable
observers. Before (or while) replacing `Cache_this`, ask:

> Which intermediate numerical results are logically part of the conditioned
> posterior, and which are merely temporary implementation details?

That question is mathematical, not merely mechanical. Stage 2N's
`ConditionedState` experiment should inform whether GPy's observer caches are
largely compensating for a missing explicit posterior-state abstraction.
Stage 3 may still ship a version-counter interim; the successor can then own
posterior-scoped caches.

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
- Introducing JAX or any autodiff dependency into the GPy 1 *runtime* (Stage 2T
  experiments may use JAX outside the package dependency graph).
- Making the entire GPy 1 codebase JAX-compatible.
- Deciding that GPy 2 must use JAX or support multiple numerical backends
  (that decision belongs to a successor architecture CIP, informed by
  CIP-0010).
- Adopting GPyTorch's full linear-operator architecture, or any JAX module
  ecosystem (Flax / Equinox / paramax / …), into GPy 1 production code.
- Cython changes (see Open questions).

### Shared implementation guidance

These principles govern Stage 2 / 2T and are shared with
[CIP-0010](cip0010.md):

1. **Improve GPy 1 first.** Every change to production GPy 1 should have an
   independent justification in correctness, testability, maintainability,
   performance or clarity. Future JAX compatibility is a desirable secondary
   consequence, not a sufficient reason to regress the existing
   implementation.
2. **Preserve analytical gradients.** Hand-derived derivatives remain available
   as numerical references and potentially as sources for custom
   differentiation rules. Autodifferentiation is evaluated against them rather
   than assumed to supersede them.
3. **Do not introduce a universal array (or operator) abstraction
   prematurely.** Simple numerical functions may be portable between NumPy
   and JAX, but inference algorithms often require backend-specific control
   flow, linear algebra, precision handling and differentiation. Stage 2N
   investigates the smallest useful covariance / posterior / parameter
   protocols experimentally — it does not mandate a production operator
   hierarchy.
4. **Separate the production and experimental environments.** GPy 1 should not
   acquire a JAX runtime dependency. Experimental JAX implementations may live
   in a separate test or research environment and need not be distributed with
   the GPy package.
5. **Treat incompatibilities as findings.** Failure to compile or differentiate
   an algorithm is useful information. Do not modify production algorithms
   merely to make every transformation test pass. Document the obstacle, its
   implications and the available alternatives.
6. **Keep GPy 2's architectural choice open.** The programme provides evidence
   about the costs and benefits of different execution models. It should not
   establish a de facto JAX commitment before the successor architecture has
   been reviewed.

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
   likelihoods. Backlog task per family. Prefer transformation-ready
   signatures where practical (see Stage 2 design principle).
5. **Stage 2T (incremental):** after the first pure LML (or similar)
   extraction, add an out-of-package JAX experiment; record barrier
   classifications and feed CIP-0010's transformation-readiness report.
   Further cases follow as Stage 2 coverage grows — not as a gate on Stage 2
   completion.
6. **Stage 2N (light-touch):** with the first exact-GP extraction, sketch the
   conceptual `CovarianceOperator` / `ConditionedState` / `ParameterSpec`
   boundaries and demonstrate condition-once / predict-many via pure state.
   Do not merge a large operator hierarchy into GPy 1.
7. **Stage 3:** caching replacement behind benchmarks; use Stage 2N findings
   to separate posterior-scoped state from temporary caches; record numbers
   in this CIP.
8. **Stage 4:** separate Accept decision with a short design note (may become
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
- Stage 2 purity checks: extracted numerical functions callable without a GPy
  model or `paramz` parameter object.
- Stage 2T (optional CI / research env): numerical and gradient parity of
  experimental JAX equivalents; `jax.jit` / `jax.vmap` probes where
  meaningful; barrier notes checked into CIP notes or a linked report (not a
  package dependency).
- Pickle round-trip and "load previous-release pickle" fixtures before Stage 3.
- Wall-clock benchmarks for Stage 3 caching changes (record in this CIP).
- Ruff `TID251` in CI enforces the Stage 0 boundary.

## Related Requirements

None yet. Candidates after Accept:

- Parameter tying across kernel components (CIP-0002)
- Pickleable models and kernels (backlog `pickle-live-kernels`)
- Numerical core reusable under functional transformations (Stage 2 / 2T;
  evidence for CIP-0010 / GPy 2 architecture)

## Implementation Status

- [x] Draft CIP-000F (Proposed)
- [x] Ideas Discussion opened ([#1293](https://github.com/SheffieldML/GPy/discussions/1293))
- [ ] Accept: stage boundaries; pickle-compatibility policy
- [ ] Stage 0: single import boundary + Ruff rule
- [ ] Stage 1: domains, optimisation vendored
- [ ] Stage 2: pure numerical functions (kernels, likelihoods);
      transformation-readiness design principle applied where practical
- [ ] Stage 2T: at least one experimental JAX assessment (LML or similar);
      barriers classified and documented for CIP-0010
- [ ] Stage 2N: conceptual numerical interfaces sketched; exact-GP
      condition-once / predict-many via pure state demonstrated
- [ ] Stage 3: caching / `ObsAr` replaced, benchmarks recorded
- [ ] Stage 4: core decision recorded
- [ ] Backlog tasks created (only after Accept)

## Open questions (resolve at Accept)

1. **CIP-0002:** fold tying into Stage 4 Option B, or keep CIP-0002 separate and
   make it depend on this CIP?
2. **Cython:** already [CIP-000E](cip000E.md)
   ([Discussion #1279](https://github.com/SheffieldML/GPy/discussions/1279)).
   This CIP does not decide keep vs remove. CIP-000E Phase 1 replacements
   should follow Stage 2 pure-function conventions where practical so the
   same hot paths are not refactored twice; successful JAX transformation is
   **not** an acceptance criterion for Cython removal. Stages 0–3 of this CIP
   do not depend on CIP-000E's Accept outcome. If CIP-000E removes the Cython
   build hook, packaging gets simpler and the paramz stages stay the same.
3. **Caching policy:** prefer posterior-scoped `ConditionedState` (Stage 2N)
   over long-lived observer caches where possible; keep a thin
   version-counter interim for GPy 1 if needed. Confirm at Accept.
4. **Stage 4 default:** A (vendor) as an interim step before B, or go to B
   directly?
5. **Operator protocol depth in GPy 1:** stop at documenting the conceptual
   boundary + dense reference implementation, or also ship a minimal
   matvec-based experimental path in-tree (still without a full hierarchy)?

## References

- [CIP-0002](cip0002.md) — Parameter tying (Option 3: replace paramz)
- [CIP-0005](cip0005.md) — Ruff adoption (`UP008` deferred for paramz)
- [CIP-0006](cip0006.md) — Serialization maintenance
- [CIP-000C](cip000C.md) — uv packaging (Cython build-hook constraint)
- [CIP-000E](cip000E.md) — Cython hygiene and direction
- [CIP-0010](cip0010.md) — GPy2 readiness (transformation-readiness report;
  evidence before backend selection)
- Backlog: `bugs/2026-10-10_pickle-live-kernels.md`,
  `infrastructure/2025-08-15_parameter-tying-framework.md`
- Tenet: [Community-Driven Development](../tenets/community-driven-development.md)
- paramz: https://github.com/sods/paramz
- Discussion: [#1293](https://github.com/SheffieldML/GPy/discussions/1293)
