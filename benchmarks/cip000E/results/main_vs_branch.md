# CIP-000E main vs branch benchmark

## Environment

- **Host:** `macOS-26.6.2-arm64-arm-64bit` (`cpu_count=8`)
- **Python / NumPy / SciPy (main):** 3.11.7 / 2.4.6 / 1.17.1
- **Python / NumPy / SciPy (branch):** 3.11.7 / 2.4.6 / 1.17.1
- **BLAS (main probe):** `accelerate`
- **Thread env:** OMP=None OPENBLAS=None VECLIB=None

## Git / defaults

| Tree | Branch | Commit | `backprop` | `flat_to_triang` | cython flags |
|---|---|---|---|---|---|
| main | `main` | `6f1f31bc` | `backprop_gradient_par_c` | `_flat_to_triang_cython` | chol=True lin=True stat=True coreg=True |
| branch | `cip000E-numpy-blas-primitives` | `4e990e11` | `cholesky_vjp_blocked` | `_flat_to_triang_idx` | chol=False lin=False stat=False coreg=False |

## Results

Times are mean wall-clock over **5** reps (same seeds). **× = branch / main** (lower means branch is faster).

| Bench | Shape | Main | Branch | × (branch/main) | Main impl | Branch impl |
|---|---|---:|---:|---:|---|---|
| `cholesky_backprop` | N=500 | 12.41ms | 31.27ms | **2.52×** | `backprop_gradient_par_c` | `cholesky_vjp_blocked` |
| `cholesky_backprop` | N=1000 | 101.75ms | 70.57ms | **0.69×** | `backprop_gradient_par_c` | `cholesky_vjp_blocked` |
| `cholesky_backprop` | N=2000 | 523.19ms | 203.95ms | **0.39×** | `backprop_gradient_par_c` | `cholesky_vjp_blocked` |
| `flat_to_triang` | M=50, D=5 | 52.0µs | 66.5µs | **1.28×** | `_flat_to_triang_cython` | `_flat_to_triang_idx` |
| `triang_to_flat` | M=50, D=5 | 11.9µs | 73.7µs | **6.20×** | `_triang_to_flat_cython` | `_triang_to_flat_idx` |
| `flat_to_triang` | M=200, D=5 | 249.1µs | 1.11ms | **4.48×** | `_flat_to_triang_cython` | `_flat_to_triang_idx` |
| `triang_to_flat` | M=200, D=5 | 122.2µs | 571.4µs | **4.68×** | `_triang_to_flat_cython` | `_triang_to_flat_idx` |
| `flat_to_triang` | M=400, D=8 | 2.29ms | 3.81ms | **1.67×** | `_flat_to_triang_cython` | `_flat_to_triang_idx` |
| `triang_to_flat` | M=400, D=8 | 2.03ms | 4.64ms | **2.28×** | `_triang_to_flat_cython` | `_triang_to_flat_idx` |
| `symmetrify` | N=500 | 286.9µs | 684.6µs | **2.39×** | `cython` | `blocked_or_numpy` |
| `symmetrify` | N=1000 | 2.41ms | 2.86ms | **1.19×** | `cython` | `blocked_or_numpy` |
| `symmetrify` | N=3000 | 16.85ms | 18.82ms | **1.12×** | `cython` | `blocked_or_numpy` |
| `gradients_X` | N=1000, M=500, Q=5 | 18.06ms | 14.69ms | **0.81×** | `cython` | `blas` |
| `lengthscale_grads` | N=1000, M=500, Q=5 | 5.15ms | 814.2µs | **0.16×** | `cython` | `blas` |
| `gradients_X` | N=3000, M=1500, Q=10 | 210.07ms | 128.30ms | **0.61×** | `cython` | `blas` |
| `lengthscale_grads` | N=3000, M=1500, Q=10 | 93.33ms | 5.34ms | **0.06×** | `cython` | `blas` |
| `coregionalize_reduce` | N=2000, D=5 | 7.19ms | 1.57ms | **0.22×** | `cython` | `blas` |
| `coregionalize_K` | N=2000, D=5 | 5.49ms | 21.75ms | **3.96×** | `cython` | `numpy` |
| `coregionalize_reduce` | N=5000, D=5 | 44.90ms | 13.62ms | **0.30×** | `cython` | `blas` |
| `coregionalize_K` | N=5000, D=5 | 39.86ms | 137.28ms | **3.44×** | `cython` | `numpy` |
| `gp_objective` | N=400, Q=3 | 9.15ms | 12.51ms | **1.37×** | `GPRegression` | `GPRegression` |
| `gp_gradients` | N=400, Q=3 | 8.20ms | 9.37ms | **1.14×** | `GPRegression` | `GPRegression` |
| `sparse_gp_objective` | N=2000, M=100, Q=4 | 21.49ms | 19.66ms | **0.92×** | `SparseGPRegression` | `SparseGPRegression` |
| `sparse_gp_gradients` | N=2000, M=100, Q=4 | 21.03ms | 19.69ms | **0.94×** | `SparseGPRegression` | `SparseGPRegression` |

## Notes

- Compares **shipped defaults**, not best-of-tree cherry-picks: `main` uses Cython where built; branch uses NumPy/BLAS primitives after CIP-000E Phase 1–2.
- Model benches (`gp_*`, `sparse_gp_*`) mix factorisation, likelihood, and kernel work — they show end-to-end impact, not a pure hotspot ratio.
- Absolute times are host-specific (Apple Accelerate on this Mac). Re-run on Linux multicore before treating ratios as universal.

