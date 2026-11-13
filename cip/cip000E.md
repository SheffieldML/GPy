---
author: Neil Lawrence
created: "2026-11-13"
id: "000E"
last_updated: "2026-11-13"
status: Proposed
compressed: false
related_requirements: []
related_cips:
- "0003"
- "0008"
- "000A"
- "000C"
tags:
- cip
- cython
- performance
- packaging
- hygiene
- state-space
title: Cython hygiene and direction (NumPy/BLAS vs compiled extensions)
---

# CIP-000E: Cython hygiene and direction (NumPy/BLAS vs compiled extensions)

## Status

- [x] Proposed - Initial idea documented (open for Discussion feedback)
- [ ] Accepted - Approved, ready to start work (Accept decisions below)
- [ ] In Progress - Actively being implemented
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

  Caveat: 2 cores limit OpenMP headroom; multithreaded BLAS would also scale
  on larger machines, so the ranking is expected to hold.
- **Cost is packaging, not speed.** Cython drives per-platform wheels, OpenMP
  flags, sdist compile failures, and CIP-000A's expensive PR matrix. If NumPy
  rewrites are faster, the build surface is paying for the wrong default.
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
| **D — New compilers for state-space only** | Numba / JAX `lax.scan` (or similar) for Kalman loops where Cython gave no speedup after the import fix. Orthogonal to A–C for the other four modules; call out as follow-on if Accept picks A or C. |

**Non-goals for this CIP (unless Accept expands them)**

- Rewriting inference algorithms beyond the five Cython-touched helpers.
- Mandating GPU / JAX as a runtime dependency of core GPy.
- Changing CIP-000C's uv cutover except to *relax* Cython build constraints
  if Option A lands.

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
  reduce, and triang maps to the benchmarked NumPy/BLAS forms.
- Keep GPy's old loop fallbacks only as long as needed for parity tests, then
  delete.
- Remove Extension list from `build_extension.py` (or delete the hook once
  CIP-000C's backend no longer needs it); drop OpenMP compile flags; shrink
  sdist (no `.pyx` required).
- Update CIP-0008 / CIP-000A assumptions: wheels may become pure-Python
  (platform tags simplify); PR Cython compile cost disappears.
- Coordinate with CIP-000C so uv/Poetry build-backend work does not invest in
  a Cython hook we are about to delete.

### Alternatives already rejected as defaults

- Leaving state-space Cython broken while advertising `use_cython=True`.
- Treating Cython as the performance story without comparing to BLAS-backed
  NumPy on current NumPy ≥ 2.

## Implementation Plan

1. **Proposed (this phase):** publish this CIP; open an Ideas Discussion with
   the benchmark table and options A–D; solicit maintainer/user input on
   multicore machines and SVGP / coregionalize workloads.
2. **Accept:** record chosen option and any Phase-0 vs rewrite sequencing in
   an Accept decisions section (CIP-000C style).
3. **Phase 0 (hygiene):** import fix + availability assertion + parity tests;
   optional under B as the whole CIP.
4. **Phase 1 (if A or C):** land NumPy/BLAS rewrites behind the existing
   dispatch (or as the only path); prove parity; bench on ≥1 multicore host.
5. **Phase 2 (if A, or C after soak):** remove Cython modules and build hook
   surface; update packaging CIPs' docs; pure-Python wheel story.
6. **Follow-on (optional D):** state-space acceleration exploration only if
   users still care after Phase 0 shows Cython ≈ Python there.

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
- Import smoke: all five extensions when built; state-space
  `cython_code_available is True` after the relative-import fix.
- Integration: SVGP / variational models exercising Cholesky parameterisation;
  coregionalized regression; a state-space likelihood/gradient path with
  `use_cython` True and False.
- Benchmarks: re-run the Claude suite (or a slimmed `GPy/testing` benchmark
  script) on a multicore Linux host before Accept closes Option A vs B.
- Packaging: if Cython removed, confirm pure-Python (or abi3) wheels and that
  CIP-0008 publish still works; if kept, keep current Cython smoke imports.

## Related Requirements

None yet. Candidates after Accept:

- Optional compiled accelerators must either load correctly or not ship.
- Hot paths should not be slower than idiomatic NumPy/BLAS on supported
  NumPy ≥ 2.
- Packaging cost of accelerators should be justified by measured speedups.

## Implementation Status

- [x] Document strategy options and hygiene scope (this CIP)
- [x] Open GitHub Discussion; link from Summary ([#1279](https://github.com/SheffieldML/GPy/discussions/1279))
- [ ] Accept decision (A / B / C / D notes)
- [ ] Phase 0 hygiene (state-space import + tests + docs)
- [ ] Phase 1 rewrites (if A or C)
- [ ] Phase 2 remove Cython build surface (if A or C soak done)
- [ ] Update CIP-0008 / 000A / 000C cross-references if build surface changes

## References

- Modules: `GPy/kern/src/stationary_cython.pyx`, `stationary_utils.c`,
  `GPy/util/choleskies_cython.pyx`, `GPy/util/linalg_cython.pyx`,
  `GPy/kern/src/coregionalize_cython.pyx`,
  `GPy/models/state_space_cython.pyx`, wrappers in matching `.py` files
- Bug locus: `GPy/models/state_space_main.py` (`import state_space_cython`)
- Build: `build_extension.py`, `pyproject.toml` `[tool.poetry.build]`
- Config: `GPy/defaults.cfg` `[cython] working`
- Related CIPs: [0003](./cip0003.md), [0008](./cip0008.md),
  [000A](./cip000A.md), [000C](./cip000C.md)
- Precedent Discussion: [#1268](https://github.com/SheffieldML/GPy/discussions/1268) (uv packaging)
- Benchmark source: Claude agent run (2026-11); scripts held outside the repo
  for now — attach or vendor under `cip/cip000E/` after Accept if we want them
  reproducible in-tree
