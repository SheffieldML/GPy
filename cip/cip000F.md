---
author: Neil Lawrence
created: "2026-11-13"
id: "000F"
last_updated: "2026-11-21"
status: In Progress
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
- [x] Accepted - Approved, ready to start work (Accept decisions below)
- [x] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

Community review: [Discussion #1293](https://github.com/SheffieldML/GPy/discussions/1293) (Ideas; same pattern as CIP-000E / [#1279](https://github.com/SheffieldML/GPy/discussions/1279)). Discussion had no replies by Accept; decisions below record the CIP defaults.

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
- Exit: **zero** direct `paramz` imports outside the two allowed packages
  (façade modules inside the boundary may exceed the original ~8 estimate;
  landed at 10 including new `caching` / `domains` re-export modules); full
  test suite unchanged.

**Stage 1 — Vendor loosely coupled pieces.** *(done 2026-11-21)*
- `domains` (constants) and `optimization` (SCG / L-BFGS-B wrappers; requires a
  GPy-owned `Model.optimize` dispatch). Landed: local domain constants;
  vendored optimiser modules under `GPy.inference.optimization`;
  `GPy.core.model.Model.optimize` / `optimize_restarts` dispatch to those
  wrappers.
- **Not** `transformations`: paramz core checks `isinstance(…, Transformation)`
  and compares `__fixed__` by identity, so these move only with the core
  (Stage 4).

**Stage 2 — Pure numerical functions.** *(acceptance met 2026-11-21; further
families may continue as follow-ups)*
- Kernels: `K` / `Kdiag` (and, where present, psi-statistics forward passes)
  delegate to module-level functions of plain `ndarray`s; parameter values are
  unwrapped once at the method boundary.
- Likelihoods: same for `logpdf` and moments.
- Kernel by kernel, with current outputs as golden values. No gradient code is
  removed in GPy 1.x.
- **Landed for acceptance:** `rbf_ops` + `stationary_ops` distances; Exponential /
  OU / Matern32 / Matern52 via `stationary_cov_ops`; Gaussian `logpdf_link` /
  `dlogpdf_dlink` via `gaussian_ops`; exact-GP LML via `exact_gp_ops` (wired
  through `ExactGaussianInference`). Golden tests:
  `test_cip000F_stage2_rbf_ops.py`, `test_cip000F_stage2_pure_numerical.py`.

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

**Seed material from CIP-000E (2026-11-16).** Cython removal already extracted
several hot-path primitives that match Stage 2 conventions (array in /
array out; no `paramz` / model coupling). Treat them as examples and as
inputs to Stage 2T, not as completion of Stage 2:

| Primitive | Location | Notes for Stage 2 / 2T |
|---|---|---|
| `cholesky_vjp`, `cholesky_vjp_blocked` | `GPy/util/choleskies.py` | Murray blocked is the default backprop; hybrid uses unblocked VJP on *updated* diagonal blocks |
| `lengthscale_grads_sqdiff`, `gradients_X_from_tmp` | `GPy/kern/src/stationary_ops.py` | Expanded BLAS forms; **mean-centre** shared ``X``/``X2`` offsets before expanding |
| `gradient_reduce` | `GPy/kern/src/coregionalize_ops.py` | One-hot / ``O.T @ dL @ O`` style |
| triang maps / blocked `symmetrify` | `choleskies.py` / `linalg.py` | Packaging win; still slower than former Cython at small sizes |

**Lessons (do not rediscover in Stage 2):**

1. **Cite the winning algorithm, then measure.** Column-matvec BLAS matched
   the oracle but lost to Cython; Murray Level-3 blocked cleared the bar.
2. **Expanded identities are not automatically numerically safe.**
   ``(x-y)^2 = x^2+y^2-2xy`` and ``Σ tmp·(x-y) = X·rowsum − tmp@X2`` fail
   under large shared offsets (e.g. Unix seconds). Shared mean-shift fixes
   the common case; direct-difference forms remain safer when the *spread*
   itself is huge vs the lengthscale.
3. **Blocked factorisations need care on stiff ``K``.** Symbolic reverse on
   a diagonal block *after* ``Dbar -= tril(Cbar.T @ C)`` is less stable than
   the unblocked recurrence on that block; keep symbolic when the whole
   factor fits in one block.
4. **JAX transformation is still not a Stage 2 / Cython-removal gate.**
   CIP-000E Accepted and removed Cython on performance, correctness and
   packaging evidence alone.

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

1. **Proposed → Accepted:** Ideas Discussion opened; Accept decisions recorded
   below (2026-11-21).
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

## Accept decisions (2026-11-21)

Accepted on branch/worktree `cip000F-paramz-reduction`
after Discussion [#1293](https://github.com/SheffieldML/GPy/discussions/1293)
(no community replies by Accept date). Stage boundaries and pickle policy as
in this CIP; open questions resolved as follows:

1. **CIP-0002:** keep **separate**. Parameter tying remains its own CIP and
   **depends on** this one (Stage 4 Option B, if chosen, is the natural
   enablement point — not a fold-in).
2. **Cython:** already resolved by CIP-000E (Option C → Phase 2 toward A;
   [Discussion #1279](https://github.com/SheffieldML/GPy/discussions/1279);
   Phase 2 merged via [#1312](https://github.com/SheffieldML/GPy/pull/1312)).
   Stage 2 **reuses** CIP-000E pure primitives (`stationary_ops`, Cholesky
   Murray, `coregionalize_ops`, …) rather than re-extracting those hotspots.
3. **Caching policy:** confirmed — prefer posterior-scoped `ConditionedState`
   (Stage 2N) over long-lived observer caches; allow a thin version-counter
   interim in GPy 1 if benchmarks require it before Stage 3 lands.
4. **Stage 4 default:** **A (vendor) as interim**, then decide on B. Keeps the
   public constrain/fix/prior/indexing surface while dropping the external
   paramz package; B stays a separate Accept with a short design note (may
   become its own CIP).
4a. **Stage 4 Accept (2026-11-21):** Option **A confirmed and implemented** —
    see [`cip/cip000F/notes/stage4-core-decision.md`](cip000F/notes/stage4-core-decision.md).
    Option B deferred (possible follow-on CIP; CIP-0002 not folded in).
5. **Operator protocol depth in GPy 1:** **conceptual boundary + dense
   reference only**. No in-tree matvec operator hierarchy in GPy 1; Stage 2N
   documents the protocols and demonstrates condition-once / predict-many via
   pure state.
6. **Pickle policy:** any stage that moves a pickled class (Stage 3 `ObsAr`;
   Stage 4 core) ships a `sys.modules` alias or custom unpickler, plus a
   load-previous-release pickle test, before merge.
7. **Worktree:** implementation proceeds on `cip000F-paramz-reduction`
   starting at Stage 0.

## Implementation Status

- [x] Draft CIP-000F (Proposed)
- [x] Ideas Discussion opened ([#1293](https://github.com/SheffieldML/GPy/discussions/1293))
- [x] Accept: stage boundaries; pickle-compatibility policy (2026-11-21)
- [x] Stage 0: single import boundary + Ruff rule
- [x] Stage 1: domains, optimisation vendored; GPy-owned `Model.optimize`
- [x] Stage 2: pure numerical functions (kernels, likelihoods);
      transformation-readiness design principle applied where practical
      (CIP-000E seeded Cholesky / stationary / coregionalize primitives —
      see Stage 2 section) — representative RBF/stationary + Gaussian LML /
      logpdf; further kernel families optional follow-ups
- [x] Stage 2T: at least one experimental JAX assessment (LML or similar);
      barriers classified and documented for CIP-0010 —
      [`cip/cip000F/notes/stage2t-jax-lml.md`](cip000F/notes/stage2t-jax-lml.md)
- [x] Stage 2N: conceptual numerical interfaces sketched; exact-GP
      condition-once / predict-many via pure state demonstrated —
      [`cip/cip000F/notes/stage2n-numerical-interfaces.md`](cip000F/notes/stage2n-numerical-interfaces.md)
- [x] Stage 3: caching / `ObsAr` replaced, benchmarks recorded —
      [`cip/cip000F/notes/stage3-caching-obsar.md`](cip000F/notes/stage3-caching-obsar.md)
- [x] Stage 4: core decision recorded — **Accept A (vendor)**;
      [`cip/cip000F/notes/stage4-core-decision.md`](cip000F/notes/stage4-core-decision.md);
      paramz core vendored under `GPy.core.parameterization._paramz`,
      external dependency dropped
- [x] Backlog tasks created (only after Accept):
      [Stage 0](../backlog/infrastructure/2026-11-21_cip000F-stage0-import-boundary.md),
      [Stage 1](../backlog/infrastructure/2026-11-21_cip000F-stage1-domains-optimisation.md),
      [Stage 2](../backlog/infrastructure/2026-11-21_cip000F-stage2-pure-numerical.md),
      [Stage 2T](../backlog/infrastructure/2026-11-21_cip000F-stage2t-jax-assessment.md),
      [Stage 2N](../backlog/infrastructure/2026-11-21_cip000F-stage2n-numerical-interfaces.md),
      [Stage 3](../backlog/infrastructure/2026-11-21_cip000F-stage3-caching-obsar.md),
      [Stage 4](../backlog/infrastructure/2026-11-21_cip000F-stage4-paramz-core-decision.md)

## Open questions (resolved at Accept)

See [Accept decisions](#accept-decisions-2026-11-21). Summaries:

1. **CIP-0002:** keep separate; depends on this CIP.
2. **Cython:** resolved by CIP-000E / [#1312](https://github.com/SheffieldML/GPy/pull/1312); reuse E primitives in Stage 2.
3. **Caching policy:** `ConditionedState` preferred; thin version-counter interim OK.
4. **Stage 4 default:** A (vendor) interim, then decide on B.
5. **Operator protocol depth:** conceptual + dense reference only in GPy 1.

## References

- [CIP-0002](cip0002.md) — Parameter tying (Option 3: replace paramz)
- [CIP-0005](cip0005.md) — Ruff adoption (`UP008` deferred for paramz)
- [CIP-0006](cip0006.md) — Serialization maintenance
- [CIP-000C](cip000C.md) — uv packaging (Cython build hook removed by CIP-000E)
- [CIP-000E](cip000E.md) — Cython hygiene and direction (Option C → A;
  pure NumPy/BLAS primitives; numerics lessons above)
- [CIP-0010](cip0010.md) — GPy2 readiness (transformation-readiness report;
  evidence before backend selection)
- Backlog: `bugs/2026-10-10_pickle-live-kernels.md`,
  `infrastructure/2025-08-15_parameter-tying-framework.md`
- Tenet: [Community-Driven Development](../tenets/community-driven-development.md)
- paramz: https://github.com/sods/paramz
- Discussion: [#1293](https://github.com/SheffieldML/GPy/discussions/1293)
