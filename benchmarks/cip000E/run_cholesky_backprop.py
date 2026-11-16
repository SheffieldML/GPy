#!/usr/bin/env python3
"""CIP-000E: Cholesky backprop runtime / memory / parity harness.

Compares:
  - pure:   GPy.util.choleskies._backprop_gradient_pure (oracle)
  - blas:   GPy.util.choleskies.cholesky_vjp (NumPy/BLAS primitive)
  - cython: choleskies_cython.backprop_gradient_par_c (if built)

Does not import JAX. Peak memory is tracemalloc (Python allocations), not RSS.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Result:
    name: str
    N: int
    seconds: float
    peak_kib: float
    max_abs_err_vs_pure: float
    reps: int


def _make_problem(N: int, seed: int):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    K = A @ A.T + float(N) * np.eye(N)
    L = np.linalg.cholesky(K)
    dL = rng.normal(size=(N, N))
    return L, dL


def _time_and_memory(fn, L, dL, reps: int) -> tuple[float, float, np.ndarray]:
    # Warmup (not traced).
    out = fn(dL, L)
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(dL, L)
    elapsed = (time.perf_counter() - t0) / reps
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024.0, out


def _env_metadata() -> dict:
    import numpy
    import scipy

    blas = {}
    try:
        blas = numpy.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    except TypeError:
        # Older NumPy
        blas = {"name": "unknown"}

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "blas": blas,
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="200,500,1000",
        help="Comma-separated matrix sizes N (default: 200,500,1000)",
    )
    parser.add_argument("--reps", type=int, default=5, help="Timed repetitions per size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit JSON summary on stdout")
    args = parser.parse_args(argv)

    from GPy.util import choleskies

    impls = {
        "pure": choleskies._backprop_gradient_pure,
        "blas": choleskies.cholesky_vjp,
    }
    try:
        from GPy.util import choleskies_cython

        impls["cython"] = lambda dL, L: np.asarray(choleskies_cython.backprop_gradient_par_c(dL, L))
    except ImportError:
        pass

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results: list[Result] = []

    for N in sizes:
        L, dL = _make_problem(N, args.seed + N)
        # Skip pure at large N (cubic Python loops).
        active = dict(impls)
        if N > 400 and "pure" in active:
            # Still compute one pure reference at small interior size for docs;
            # for large N use blas as reference only if pure omitted.
            pass
        ref_name = "pure" if "pure" in active and N <= 400 else "blas"
        # Always get a pure reference for N <= 400; for larger, trust blas vs cython.
        if N <= 400:
            ref = choleskies._backprop_gradient_pure(dL, L)
        else:
            ref = choleskies.cholesky_vjp(dL, L)

        for name, fn in active.items():
            if name == "pure" and N > 400:
                continue
            seconds, peak_kib, out = _time_and_memory(fn, L, dL, args.reps)
            err = float(np.max(np.abs(out - ref))) if name != ref_name else 0.0
            # When ref is blas and name is cython, still report error vs blas.
            if N > 400 and name == "cython":
                err = float(np.max(np.abs(out - ref)))
            results.append(
                Result(
                    name=name,
                    N=N,
                    seconds=seconds,
                    peak_kib=peak_kib,
                    max_abs_err_vs_pure=err,
                    reps=args.reps,
                )
            )

    meta = _env_metadata()
    if args.json:
        print(json.dumps({"environment": meta, "results": [asdict(r) for r in results]}, indent=2))
    else:
        print("CIP-000E Cholesky backprop benchmark")
        print("Environment:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()
        print(f"{'impl':<8} {'N':>6} {'seconds':>10} {'peak_KiB':>10} {'max_abs_err':>12}")
        for r in results:
            print(
                f"{r.name:<8} {r.N:>6d} {r.seconds:>10.4f} {r.peak_kib:>10.1f} {r.max_abs_err_vs_pure:>12.3e}"
            )
        print()
        print(
            "Barrier note (CIP-000F): cholesky_vjp is a pure numerical primitive; "
            "a future JAX custom VJP may wrap the same recurrence."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
