# CIP-000E numerical benchmarks

Reproducible harness for Cython → NumPy/BLAS replacement decisions
([CIP-000E](../../cip/cip000E.md)).

## Main vs branch (shipped defaults)

Compares the **wired public entry points** on a `main` checkout (Cython when
built) against this branch (NumPy/BLAS after Phase 1–2):

```bash
# from the cip000E branch worktree; --main-root points at a synced main tree
env -u PYTHONPATH python benchmarks/cip000E/run_main_vs_branch.py \
  --main-root /path/to/GPy-on-main \
  --reps 5 \
  --out-dir benchmarks/cip000E/results
```

Writes `results/main_vs_branch.md` and `results/main_vs_branch.json`.
Clears `PYTHONPATH` so a multi-root workspace cannot pollute imports.

## Numerical stability (CIP-0011 compare)

Adversarial outcome matrix (`pass` / `fail` / `xfail` / …), not wall-clock.
Recorded snapshot (2026-11-17): `main` vs this branch were **outcome-identical**
(0 diffs, 20 shared xfails) — see
[`results/numerical_stability_main_vs_branch.md`](results/numerical_stability_main_vs_branch.md).

```bash
# needs a checkout that has numerical_stability/ (CIP-0011 branch or merged main)
uv run python numerical_stability/compare.py \
  --baseline /path/to/GPy-on-main \
  --candidate /path/to/GPy-cip000E \
  --suite-from /path/to/checkout-with-suite \
  --out /tmp/gpy-num-compare-main-vs-cythonless
```

Defect fixes that clear those xfails are CIP-0012, not this CIP.

## Per-hotspot harnesses (single tree)

```bash
# from repo root, with the project env active
python benchmarks/cip000E/run_cholesky_backprop.py
python benchmarks/cip000E/run_cholesky_backprop.py --sizes 500,1000,2000 --reps 5
python benchmarks/cip000E/run_stationary_grads.py --N 3000 --Q 10
python benchmarks/cip000E/run_coregionalize_reduce.py --N 5000 --D 5
python benchmarks/cip000E/bench_suite.py --label local --reps 5
```

Single-tree harnesses report implementations available in *this* install
(`pure` / `blas` / `murray` / `cython` where present).

Metrics: wall-clock (mean), relative time vs Cython or vs main, peak traced
Python allocations (`tracemalloc`), plus NumPy/SciPy / BLAS / thread-env
metadata. Model benches bust the paramz cache each call.

**Successful JAX transformation is not required** for Cython removal; these
harnesses only record NumPy/Cython evidence for Accept / Phase 2.
