## Main vs branch benchmarks (2026-11-16)

Fair comparison of **shipped defaults** on the same Mac (arm64 / Apple Accelerate, Python 3.11.7, NumPy 2.4.6, SciPy 1.17.1, 5 reps):

| Tree | Ref | Cholesky backprop default | Cython extensions |
|---|---|---|---|
| `main` | `6f1f31bc` | Cython `backprop_gradient_par_c` | built / enabled |
| `cip000E-numpy-blas-primitives` | `87ccbfb7` | Murray `cholesky_vjp_blocked` | removed (Phase 2) |

**× = branch / main** (lower ⇒ branch faster):

| Hotspot | Shape | × |
|---|---|---|
| Cholesky backprop | N=500 / 1000 / 2000 | **0.46 / 0.28 / 0.23** |
| `lengthscale_grads` | N=1000 / 3000 | **0.11 / 0.06** |
| `gradients_X` | N=1000 / 3000 | **0.78 / 0.62** |
| coregionalize `gradient_reduce` | N=2000 / 5000 | **0.21 / 0.31** |
| SparseGP objective / gradients | N=2000, M=100 | **0.98 / 0.93** |
| Dense GP objective / gradients | N=400 | 1.39 / 1.47 |
| triang maps (`flat`↔`triang`) | M=50…400 | 2–12× slower (µs–ms absolute) |
| `symmetrify` | N=500 / 1000 / 3000 | 5.6 / 1.9 / 1.4× slower |
| coregionalize `K` | N=2000 / 5000 | ~3× slower |

### Takeaway

The dominant hotspots that justified keeping Cython (Cholesky VJP, stationary grads, coregionalize reduce) are **faster** on the NumPy/BLAS branch. End-to-end SparseGP is parity-or-better. Remaining regressions are small absolute helpers or indexing (`symmetrify`, triang maps, `coregionalize.K`) accepted for the packaging win (pure-Python wheels / no compiler).

### Reproduce

In-repo harness (clears `PYTHONPATH` so multi-root workspaces cannot pollute imports):

```bash
env -u PYTHONPATH python benchmarks/cip000E/run_main_vs_branch.py \
  --main-root /path/to/GPy-on-main \
  --reps 5 \
  --out-dir benchmarks/cip000E/results
```

Artefacts: `benchmarks/cip000E/results/main_vs_branch.md` (+ `.json`). Full write-up in [CIP-000E](https://github.com/SheffieldML/GPy/blob/cip000E-numpy-blas-primitives/cip/cip000E.md). Linux multicore re-run still welcome.
