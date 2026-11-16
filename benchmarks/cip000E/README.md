# CIP-000E numerical benchmarks

Reproducible harness for Cython → NumPy/BLAS replacement decisions
([CIP-000E](../../cip/cip000E.md)).

## Cholesky backpropagation

```bash
# from repo root, with the project env active
python benchmarks/cip000E/run_cholesky_backprop.py
python benchmarks/cip000E/run_cholesky_backprop.py --sizes 500,1000,2000 --reps 5
```

Reports for each implementation:

- `pure` — triple-loop oracle (small ``N`` only)
- `vjp` — column BLAS matvecs (`cholesky_vjp`)
- `murray` — Murray Level-3 blocked (`cholesky_vjp_blocked`, arXiv:1602.07527)
- `cython` — `backprop_gradient_par_c` if built

Metrics: wall-clock (mean), relative time vs Cython, peak traced Python
allocations (`tracemalloc`), max abs error vs reference, plus NumPy/SciPy /
BLAS / thread-env metadata.

**Successful JAX transformation is not required** for Cython removal; this
harness only records NumPy/Cython evidence for Accept.
