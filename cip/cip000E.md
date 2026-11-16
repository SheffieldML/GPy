---
author: Neil Lawrence
created: "2026-11-13"
id: "000E"
last_updated: "2026-11-16"
status: In Progress
compressed: false
related_requirements: []
related_cips:
- "0003"
- "0008"
- "000A"
- "000C"
- "000F"
- "0010"
tags:
- cip
- cython
- performance
- packaging
- hygiene
- state-space
- transformation-readiness
title: Cython hygiene and direction (NumPy/BLAS vs compiled extensions)
---

# CIP-000E: Cython hygiene and direction (NumPy/BLAS vs compiled extensions)

## Status

- [x] Proposed - Initial idea documented (open for Discussion feedback)
- [x] Accepted - Approved, ready to start work (Accept decisions below)
- [x] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

Community review: [Discussion #1279](https://github.com/SheffieldML/GPy/discussions/1279) (Ideas; same pattern as CIP-000C / [#1268](https://github.com/SheffieldML/GPy/discussions/1268)).

## Summary

Decide GPy's medium-term stance on **Cython**, then execute the matching
hygiene (and optional removal or rewrite) work.

GPy currently ships five optional Cython modules as hot-loop speed-ups with
NumPy fallbacks. Packaging (CIP-0003), release wheels (CIP-0008), PR matrix
cost (CIP-000A), and the uv cutover (CIP-000C) all treat that Cython surface as
a hard constraint. Fresh benchmarks suggest **modern NumPy/BLAS rewrites beat
those Cython paths in most cases**, while the state-space Cython extension
builds but never loads because of an absolute-import bug.

This CIP has two layers:

1. **Strategy (Discussion first):** keep Cython, replace with NumPy/BLAS, or
   remove compiled extensions entirely (with a separate story for state-space).
2. **Hygiene (safe under any strategy):** fix the state-space import, document
   the config/`use_cython` split, retire dead helpers, and land numerical
   parity tests for fallback vs fast paths.

**Driving priority:** Cython replacement is justified by **immediate
performance, correctness and maintenance** benefits. Where practical,
replacements should also be extracted as **clean numerical primitives**
(CIP-000F Stage 2 conventions) so they do not recreate architectural barriers
that GPy 2 work would later have to remove. Making every replacement
JAX-compatible is **not** required — JAX may later probe those primitives
under CIP-000F Stage 2T / CIP-0010 Workstream G; it must not dictate the
initial NumPy/BLAS implementation.

## Motivation

- **Dead state-space Cython.** `GPy/models/state_space_main.py` uses
  `import state_space_cython` (absolute). Inside the installed package that
  raises `ModuleNotFoundError`, so `cython_code_available` is always `False`
  and `use_cython=True` quietly falls back to Python. Confirmed on a Poetry
  env with the `.so` present: package import works; absolute import fails.
- **Speed claim is stale.** Claude benchmarking (Python 3.13, NumPy 2.5,
  2-core Linux) compared Cython to GPy's own fallbacks and to rewritten
  NumPy/BLAS forms (parity-checked against GPy). Summary (× = time relative
  to Cython; lower is better):

  | Function | Size | GPy NumPy fallback | NumPy/BLAS rewrite |
  |---|---|---|---|
  | Cholesky backprop | N=1000 / 2000 | ~600× slower at N=300 (Python loops) | **0.15× / 0.07×** (Murray blocked) |
  | `lengthscale_grads` | N=3000, Q=10 | 4.2× | **0.15×** (one matmul) |
  | `gradients_X` | N=3000, Q=10 | 6.3× | **0.28×** (`X * rowsum - tmp @ X2`) |
  | coregionalize `gradient_reduce` | N=5000, D=5 | 2.8× | **0.6×** |
  | coregionalize `K` | N=5000 | ~1× | fallback already ties |
  | `flat_to_triang` / `triang_to_flat` | M=200 | 80–90× | **~1.1×** (`tril_indices`) |
  | `symmetrify` | N=1000 / 3000 | 7–9× | 2–3× / ~0.6× |
  | Kalman (state-space, import fixed locally) | N=2000 | — | Cython ≈ Python (~765 vs 759 ms) |

  Caveat: preliminary results from a limited 2-core machine; the original
  scripts were out-of-repo. An in-repo harness now lives at
  `benchmarks/cip000E/`. **Reproduced 2026-11-16 on macOS arm64 / Apple
  Accelerate (8 cores):** Murray blocked (`cholesky_vjp_blocked`,
  $N_b{=}128$) vs Cython `backprop_gradient_par_c` — × relative to Cython:

  | N | Murray ×cython | column-`vjp` ×cython |
  |---|---|---|
  | 500 | **0.46×** | 3.9× |
  | 1000 | **0.26×** | 4.1× |
  | 2000 | **0.23×** | 1.7× |

  So the win is **not** an artefact of a weak BLAS: Accelerate is strong on
  this host, and the Level-3 blocked form beats Cython here too (less
  dramatic than the early Linux 0.15×/0.07× table, but clearly faster). The
  column-matvec `cholesky_vjp` does *not* beat Cython on this Mac — do not
  confuse the two primitives. Multithreaded BLAS / OpenMP rankings can still
  shift; **test threading, do not assume it**. See
  [Reproducible benchmark requirement](#reproducible-benchmark-requirement)
  before Accept closes Option A vs B.
- **Cost is packaging, not speed.** Cython drives per-platform wheels, OpenMP
  flags, sdist compile failures, and CIP-000A's expensive PR matrix. If NumPy
  rewrites are faster, the build surface is paying for the wrong default.
- **These hotspots sit on the model–derivative boundary.** Cholesky
  backpropagation, stationary kernel gradients, coregionalisation reductions
  and triangular parameter maps are not arbitrary loops — they are exactly
  the primitives CIP-000F Stage 2 / 2T cares about. Replacing them is an
  opportunity to establish cleaner numerical boundaries, not only to win
  wall-clock.
- **Hygiene debt regardless of strategy.** Unused `cholupdate` in
  `linalg_cython.pyx`; unused `prange` / OpenMP helpers beside the live
  stationary call path; two enable mechanisms (`[cython] working` vs
  per-model `use_cython`).

## Detailed Description

### Current surface (five modules)

| Module | Role | Enable path |
|--------|------|-------------|
| `GPy.kern.src.stationary_cython` (+ `stationary_utils.c`) | ARD lengthscale grads, `gradients_X` | `config [cython] working` |
| `GPy.util.choleskies_cython` | Flat/triang Cholesky maps; backprop | same |
| `GPy.util.linalg_cython` | `symmetrify` (live); `cholupdate` (unused) | same |
| `GPy.kern.src.coregionalize_cython` | `K_*`, `gradient_reduce` | same |
| `GPy.models.state_space_cython` | Kalman / smoother classes | `StateSpace(..., use_cython=)` — **import broken** |

Build: Poetry hook `build_extension.py` (`generate-setup-file = true`);
Linux `-fopenmp -O3` + `-lgomp`; macOS `-O3` only; Cython ≥ 0.29 build-only
(CIP-0003).

### Strategic options (Decide in Discussion → Accept)

| Option | Verdict framing |
|--------|-----------------|
| **A — NumPy/BLAS default; drop Cython** | Preferred candidate from benchmarks. Replace hot paths with vectorised/BLAS code; remove `.pyx` / Extension build; pure-Python (plus NumPy/SciPy) package. Biggest win for CIP-0008/000A/000C complexity. |
| **B — Keep Cython; fix hygiene only** | Minimal change. Fix state-space import; parity tests; document config. Keeps wheel/OpenMP cost; does not capture rewrite speedups. |
| **C — Hybrid** | Land NumPy/BLAS rewrites as the default path; keep Cython behind config for one release as escape hatch; delete Cython in a follow-on CIP once telemetry/tests are green. |
| **D — New compilers for state-space only** | Numba / JAX `lax.scan` (or similar) for Kalman loops where Cython gave no speedup after the import fix. Orthogonal to A–C for the other four modules; call out as follow-on if Accept picks A or C. Do **not** couple Option D into Phase 1 Cython removal — treat functional state-transition redesign as a separately measured task. |

### Per-area treatment (do not force one transformation strategy)

| Existing Cython area | Immediate priority | Future architectural relevance |
|---|---|---|
| Cholesky derivatives | Fast, stable NumPy/BLAS replacement; prefer an explicit derivative primitive (e.g. conceptual `cholesky_vjp(L, cotangent)`) over embedding in parameter-update machinery | Very high — derivative rules and factorisation |
| Stationary kernel derivatives | Vectorise / BLAS; watch peak memory of \(N \times M \times Q\) intermediates | High — pure kernels, broadcasting, autodiff parity |
| Coregionalisation | Efficient array operations | High — multi-output shape semantics |
| Triangular matrix helpers | Simplify implementation and packaging | Moderate — parameter transforms and memory layout |
| State-space Kalman routines | Fix broken import; establish reliable baseline; drop Cython if no measured benefit | Very high — sequential computation; future `lax.scan`-style form is a *separate* redesign |

**Vectorisation warning:** eliminating Python loops via NumPy broadcasting is
not automatically transformation-ready. Large temporaries can win runtime and
still hurt memory (and later JAX). Prefer formulations based on established
linear algebra operations where those are demonstrably efficient.

### Design principle — Replace Cython with reusable numerical primitives

Cython replacement is primarily justified by measured performance, numerical
correctness, maintainability and reduced packaging complexity.

However, the replacement should also support the architectural direction
established by [CIP-000F](cip000F.md) Stage 2 and [CIP-0010](cip0010.md).

Where practical, NumPy/BLAS replacements should:

- Expose pure numerical functions with explicit array and parameter inputs
  and explicit outputs.
- Avoid coupling numerical primitives to `paramz`, model observers or
  implicit cache invalidation.
- Avoid mutation of externally owned arrays, while permitting local mutation
  where justified by performance.
- Prefer efficient linear algebra formulations over unnecessary
  high-dimensional broadcasting.
- Preserve existing analytical derivatives and numerical parity tests.
- Document shape, dtype, memory-layout and precision assumptions.
- Record operations that would require a different formulation or custom
  differentiation rule under JAX.

These requirements should be applied **proportionately**. They must not
introduce a JAX runtime dependency or compromise GPy 1 performance merely to
achieve compatibility with future program transformations.

### Coordination with CIP-000F

| CIP | Owns |
|---|---|
| **CIP-000E** | Replacement, performance validation and retirement of existing Cython extensions |
| **CIP-000F** | Broader numerical architecture, parameter decoupling and transformation-readiness experiments |

When replacing a Cython operation, CIP-000E should adopt the pure numerical
function conventions of CIP-000F Stage 2 where practical, **avoiding a second
refactoring of the same operation** (Cython → stateful NumPy inside GPy
objects → pure functions later). Prefer:

**Cython → pure, efficient NumPy/BLAS primitive**; existing GPy API delegates
to the primitive; experimental JAX code (Stage 2T) can assess it independently.

The resulting numerical primitives may be used by CIP-000F Stage 2T as
reference implementations for experimental JAX differentiation, compilation
or vectorisation tests.

**Successful JAX transformation is not an acceptance criterion for Cython
removal.**

### Hygiene work (valid under B, and as Phase 0 under A/C)

1. Change `state_space_main.py` to `from . import state_space_cython`.
2. Add a smoke test that `GPy.models.state_space_main.cython_code_available`
   is True when the extension is built (and document the `use_cython=True`
   path).
3. Document `[cython] working` vs `StateSpace(use_cython=...)` in contributor
   docs / CIP Accept notes.
4. Mark or remove dead `cholupdate` and unused stationary helpers, or wire
   them only if benchmarks justify keeping them.
5. Extend `GPy/testing/test_cython.py` (and state-space tests) so Cython and
   NumPy paths stay numerically aligned.

### If Accept chooses A or C (rewrite / remove)

- Port Cholesky backprop (largest win), stationary grads, coregionalize
  reduce, and triang maps to NumPy/BLAS forms that follow the pure-function
  conventions above (Stage 2 where practical).
- For Cholesky backpropagation, investigate an explicitly defined derivative
  primitive rather than embedding the replacement in parameter-update
  machinery; keep it independent of GPy model objects / `paramz` / mutation
  of externally owned gradient buffers.
- Keep GPy's old loop fallbacks only as long as needed for parity tests, then
  delete.
- Remove Extension list from `build_extension.py` (or delete the hook once
  CIP-000C's backend no longer needs it); drop OpenMP compile flags; shrink
  sdist (no `.pyx` required).
- Update CIP-0008 / CIP-000A assumptions: wheels may become pure-Python
  (platform tags simplify); PR Cython compile cost disappears.
- Coordinate with CIP-000C so uv/Poetry build-backend work does not invest in
  a Cython hook we are about to delete.
- For state-space: distinguish **immediate Cython removal / import hygiene**
  from any subsequent redesign around explicit functional state transitions
  \((s_{t+1},\ell_t)=f(s_t,y_t,\theta)\). The latter is independently
  justified and measured (Option D / follow-on), not forced into the first
  removal PR.

### Additional performance and verification requirements

For each replacement:

1. Benchmark runtime across representative small, medium and large input sizes.
2. Measure **peak memory** and significant intermediate allocations
   (especially stationary-kernel gradients and Cholesky derivatives).
3. Verify outputs and gradients against the existing implementation and
   independent numerical or finite-difference references.
4. Confirm performance under representative **BLAS threading** configurations.
5. Document numerical tolerances and precision assumptions.
6. Record whether the replacement has introduced or eliminated architectural
   barriers relevant to CIP-000F (mutation, layout, dynamically sized arrays,
   custom-derivative needs).

### Reproducible benchmark requirement

Before selecting Option A, B or C, commit or otherwise publish the benchmark
scripts, input generators, dependency versions, hardware details and
correctness checks (e.g. under `cip/cip000E/` or `benchmarks/`).

The existing preliminary results are promising but should **not** be the sole
evidence for irreversible removal decisions. In particular, the claim that
multithreaded BLAS preserves the observed speed advantage must be tested.

### Alternatives already rejected as defaults

- Leaving state-space Cython broken while advertising `use_cython=True`.
- Treating Cython as the performance story without comparing to BLAS-backed
  NumPy on current NumPy ≥ 2.
- Cython → fast NumPy code still tightly coupled to stateful GPy / `paramz`
  objects, then a second CIP-000F refactor of the same hot path.

### Non-goals

- Requiring every NumPy replacement to compile under `jax.jit`.
- Introducing JAX, Numba or another compiler into GPy 1 as a consequence of
  Cython removal (Option D remains a separate, measured follow-on).
- Creating a generic numerical backend abstraction.
- Rewriting full inference algorithms beyond the existing Cython replacement
  scope.
- Delaying demonstrated GPy 1 performance improvements until GPy 2
  architecture decisions are complete.
- Changing CIP-000C's uv cutover except to *relax* Cython build constraints
  if Option A lands.

### Outcome

CIP-000E should deliver faster, simpler and more maintainable GPy 1 numerical
code while producing reusable numerical primitives and evidence that informs,
but does not constrain, future GPy 2 architecture decisions.

## Implementation Plan

1. **Proposed (this phase):** publish this CIP; open an Ideas Discussion with
   the benchmark table and options A–D; solicit maintainer/user input on
   multicore machines and SVGP / coregionalize workloads. Align wording with
   CIP-000F / CIP-0010 (this amendment).
2. **Reproduce benchmarks in-tree** (or publish harness) before Accept closes
   A vs B; include peak memory and BLAS-threading variants.
3. **Accept:** record chosen option and any Phase-0 vs rewrite sequencing in
   an Accept decisions section (CIP-000C style); note CIP-000F Stage 2
   conventions for Phase 1 primitives.
4. **Phase 0 (hygiene):** import fix + availability assertion + parity tests;
   optional under B as the whole CIP.
5. **Phase 1 (if A or C):** land pure NumPy/BLAS primitives; GPy wrappers
   delegate; prove parity; bench runtime + peak memory on ≥1 multicore host;
   record CIP-000F barrier notes for each replaced op.
6. **Phase 2 (if A, or C after soak):** remove Cython modules and build hook
   surface; update packaging CIPs' docs; pure-Python wheel story.
7. **Follow-on (optional D / state-space redesign):** only if users still care
   after Phase 0 shows Cython ≈ Python; explicit state-transition form
   measured separately from Cython removal.

## Backward Compatibility

- **Option B:** behaviour change only for users who pass `use_cython=True` and
  have a built extension — they finally get Cython state-space (same numerics
  expected; no speedup expected from current evidence).
- **Option A / late C:** no Cython binary extensions; `[cython] working` and
  `use_cython` become no-ops or are removed with a deprecation window.
  Numerics should match within existing test tolerances.
- **Install:** Option A simplifies `pip install gpy` (fewer compile failures
  from sdists). Wheel matrix may shrink.

## Testing Strategy

- Unit parity: Cython vs NumPy (and rewrite vs both) for choleskies,
  stationary grads, coregionalize, symmetrify; finite-difference checks where
  already used in the benchmark work.
- Purity checks (Phase 1): replaced primitives callable without a GPy model
  or `paramz` parameter object where Stage 2 conventions apply.
- Import smoke: all five extensions when built; state-space
  `cython_code_available is True` after the relative-import fix.
- Integration: SVGP / variational models exercising Cholesky parameterisation;
  coregionalized regression; a state-space likelihood/gradient path with
  `use_cython` True and False.
- Benchmarks: published harness; runtime across sizes; peak memory;
  representative BLAS threading; multicore host before Accept closes
  Option A vs B.
- Optional (not acceptance): CIP-000F Stage 2T may probe selected primitives
  under JAX; failures are findings, not CIP-000E blockers.
- Packaging: if Cython removed, confirm pure-Python (or abi3) wheels and that
  CIP-0008 publish still works; if kept, keep current Cython smoke imports.

## Related Requirements

None yet. Candidates after Accept:

- Optional compiled accelerators must either load correctly or not ship.
- Hot paths should not be slower than idiomatic NumPy/BLAS on supported
  NumPy ≥ 2.
- Packaging cost of accelerators should be justified by measured speedups.
- Numerical hot-path replacements should be reusable primitives where
  practical (feeds CIP-000F / CIP-0010 evidence, without JAX as a runtime
  requirement).


## Accept decisions (2026-11-16)

Accepted on worktree `cip000E-numpy-blas-primitives` after in-repo harness
evidence (macOS arm64 / Apple Accelerate) and Discussion [#1279](https://github.com/SheffieldML/GPy/discussions/1279).

1. **Option C — Hybrid.** NumPy/BLAS rewrites become the default path;
   Cython remains available for one soak window as an escape hatch / residual
   accelerator, then Phase 2 removes the build surface (toward Option A).
2. **Phase 0** is already done on `main` (state-space relative import + smoke
   test; [#1302](https://github.com/SheffieldML/GPy/pull/1302)).
3. **Phase 1 starts with Cholesky backprop:** default
   `GPy.util.choleskies.backprop_gradient` is Murray blocked
   (`cholesky_vjp_blocked`, arXiv:1602.07527). Column-matvec `cholesky_vjp`
   stays as a simpler primitive/reference. Cython
   `backprop_gradient_par_c` remains importable for A/B timing, not the
   wired default. Flat/triang maps and other Cython modules stay on their
   current defaults until measured replacements land.
4. **CIP-000F Stage 2:** Phase 1 replacements are pure numerical primitives
   (array in / array out; no `paramz` coupling). Successful JAX transformation
   is **not** an acceptance criterion for Cython removal.
5. **Remaining Phase 1 order:** stationary grads → coregionalize reduce →
   triang maps → `symmetrify`, each with harness evidence before flipping
   that default. Multicore Linux + BLAS-threading confirmation before Phase 2.
6. **Option D** deferred: state-space functional redesign only if still needed
   after Phase 0 evidence (Cython ≈ Python there).

## Implementation Status

- [x] Document strategy options and hygiene scope (this CIP)
- [x] Open GitHub Discussion; link from Summary ([#1279](https://github.com/SheffieldML/GPy/discussions/1279))
- [x] Align with CIP-000F / CIP-0010 (pure-primitive replacements; JAX not an
      acceptance criterion for Cython removal)
- [x] Publish reproducible benchmark harness (`benchmarks/cip000E/`; Cholesky
      first; extend per hotspot)
- [x] Accept decision: **Option C** (see Accept decisions above)
- [x] Phase 0 hygiene (state-space import + tests; [#1302](https://github.com/SheffieldML/GPy/pull/1302))
- [ ] Phase 1 pure NumPy/BLAS primitives + wrappers — Cholesky default flipped;
      stationary / coregionalize / triang / symmetrify still pending
- [ ] Phase 2 remove Cython build surface (after soak)
- [ ] Update CIP-0008 / 000A / 000C cross-references if build surface changes
- [ ] (Optional follow-on) State-space functional state-transition redesign /
      Option D — separately measured

## References

- Modules: `GPy/kern/src/stationary_cython.pyx`, `stationary_utils.c`,
  `GPy/util/choleskies_cython.pyx`, `GPy/util/linalg_cython.pyx`,
  `GPy/kern/src/coregionalize_cython.pyx`,
  `GPy/models/state_space_cython.pyx`, wrappers in matching `.py` files
- Bug locus: `GPy/models/state_space_main.py` (`import state_space_cython`)
- Build: `build_extension.py`, `pyproject.toml` `[tool.poetry.build]`
- Config: `GPy/defaults.cfg` `[cython] working`
- Related CIPs: [0003](./cip0003.md), [0008](./cip0008.md),
  [000A](./cip000A.md), [000C](./cip000C.md),
  [000F](./cip000F.md) (Stage 2 pure functions / 2T experiments),
  [0010](./cip0010.md) (GPy2 readiness; transformation-readiness report)
- Precedent Discussion: [#1268](https://github.com/SheffieldML/GPy/discussions/1268) (uv packaging)
- Discussion: [#1279](https://github.com/SheffieldML/GPy/discussions/1279)
- Benchmark source: Claude agent run (2026-11); scripts held outside the repo
  for now — **must** be committed or published before Accept closes Option A
  vs B (see Reproducible benchmark requirement)
