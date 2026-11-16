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
| branch | `cip000E-numpy-blas-primitives` | `87ccbfb7` | `cholesky_vjp_blocked` | `_flat_to_triang_idx` | chol=False lin=False stat=False coreg=False |

## Results

Times are mean wall-clock over **5** reps (same seeds). **× = branch / main** (lower means branch is faster).

| Bench | Shape | Main | Branch | × (branch/main) | Main impl | Branch impl |
|---|---|---:|---:|---:|---|---|
| `cholesky_backprop` | N=500 | 9.86ms | 4.52ms | **0.46×** | `backprop_gradient_par_c` | `cholesky_vjp_blocked` |
| `cholesky_backprop` | N=1000 | 65.70ms | 18.68ms | **0.28×** | `backprop_gradient_par_c` | `cholesky_vjp_blocked` |
| `cholesky_backprop` | N=2000 | 534.79ms | 122.47ms | **0.23×** | `backprop_gradient_par_c` | `cholesky_vjp_blocked` |
| `flat_to_triang` | M=50, D=5 | 30.4µs | 155.1µs | **5.11×** | `_flat_to_triang_cython` | `_flat_to_triang_idx` |
| `triang_to_flat` | M=50, D=5 | 11.9µs | 142.7µs | **12.01×** | `_triang_to_flat_cython` | `_triang_to_flat_idx` |
| `flat_to_triang` | M=200, D=5 | 262.2µs | 1.10ms | **4.20×** | `_flat_to_triang_cython` | `_flat_to_triang_idx` |
| `triang_to_flat` | M=200, D=5 | 124.6µs | 960.7µs | **7.71×** | `_triang_to_flat_cython` | `_triang_to_flat_idx` |
| `flat_to_triang` | M=400, D=8 | 2.00ms | 4.62ms | **2.32×** | `_flat_to_triang_cython` | `_flat_to_triang_idx` |
| `triang_to_flat` | M=400, D=8 | 1.61ms | 4.61ms | **2.86×** | `_triang_to_flat_cython` | `_triang_to_flat_idx` |
| `symmetrify` | N=500 | 254.8µs | 1.42ms | **5.58×** | `cython` | `blocked_or_numpy` |
| `symmetrify` | N=1000 | 2.01ms | 3.90ms | **1.94×** | `cython` | `blocked_or_numpy` |
| `symmetrify` | N=3000 | 16.77ms | 23.70ms | **1.41×** | `cython` | `blocked_or_numpy` |
| `gradients_X` | N=1000, M=500, Q=5 | 19.09ms | 14.96ms | **0.78×** | `cython` | `blas` |
| `lengthscale_grads` | N=1000, M=500, Q=5 | 5.19ms | 565.2µs | **0.11×** | `cython` | `blas` |
| `gradients_X` | N=3000, M=1500, Q=10 | 211.85ms | 130.36ms | **0.62×** | `cython` | `blas` |
| `lengthscale_grads` | N=3000, M=1500, Q=10 | 92.88ms | 5.81ms | **0.06×** | `cython` | `blas` |
| `coregionalize_reduce` | N=2000, D=5 | 7.25ms | 1.50ms | **0.21×** | `cython` | `blas` |
| `coregionalize_K` | N=2000, D=5 | 6.62ms | 22.28ms | **3.36×** | `cython` | `numpy` |
| `coregionalize_reduce` | N=5000, D=5 | 44.97ms | 14.12ms | **0.31×** | `cython` | `blas` |
| `coregionalize_K` | N=5000, D=5 | 45.04ms | 133.22ms | **2.96×** | `cython` | `numpy` |
| `gp_objective` | N=400, Q=3 | 8.66ms | 12.06ms | **1.39×** | `GPRegression` | `GPRegression` |
| `gp_gradients` | N=400, Q=3 | 8.63ms | 12.73ms | **1.47×** | `GPRegression` | `GPRegression` |
| `sparse_gp_objective` | N=2000, M=100, Q=4 | 20.35ms | 19.88ms | **0.98×** | `SparseGPRegression` | `SparseGPRegression` |
| `sparse_gp_gradients` | N=2000, M=100, Q=4 | 21.57ms | 20.09ms | **0.93×** | `SparseGPRegression` | `SparseGPRegression` |

## Notes

- Compares **shipped defaults**, not best-of-tree cherry-picks: `main` uses Cython where built; branch uses NumPy/BLAS primitives after CIP-000E Phase 1–2.
- Model benches (`gp_*`, `sparse_gp_*`) mix factorisation, likelihood, and kernel work — they show end-to-end impact, not a pure hotspot ratio.
- Absolute times are host-specific (Apple Accelerate on this Mac). Re-run on Linux multicore before treating ratios as universal.

