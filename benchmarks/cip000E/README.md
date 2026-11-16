# CIP-000E numerical benchmarks

Reproducible harness for Cython → NumPy/BLAS replacement decisions
([CIP-000E](../../cip/cip000E.md)).

## Cholesky backpropagation

```bash
# from repo root, with the project env active
python benchmarks/cip000E/run_cholesky_backprop.py
python benchmarks/cip000E/run_cholesky_backprop.py --sizes 500,1000,2000 --reps 5
```

Reports for each implementation (`pure`, `blas` / `cholesky_vjp`, `cython` if built):

- wall-clock time (mean over reps)
- peak traced Python allocations (`tracemalloc`)
- max abs error vs the pure triple-loop oracle
- environment: NumPy/SciPy versions, BLAS, CPU count, thread env vars

**Successful JAX transformation is not required** for Cython removal; this
harness only records NumPy/Cython evidence for Accept.
