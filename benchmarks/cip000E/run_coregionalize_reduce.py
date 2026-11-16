#!/usr/bin/env python3
"""CIP-000E: coregionalize gradient_reduce harness."""

from __future__ import annotations

import argparse
import os
import platform
import time
import tracemalloc

import numpy as np


def _env():
    import numpy
    import scipy

    try:
        blas = numpy.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    except TypeError:
        blas = {"name": "unknown"}
    return {
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "blas": blas,
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    }


def _time(fn, reps):
    fn()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    elapsed = (time.perf_counter() - t0) / reps
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024.0, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--D", type=int, default=5)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from GPy.kern.src import coregionalize_ops

    try:
        from GPy.kern.src import coregionalize_cython

        have_cython = True
    except ImportError:
        have_cython = False

    rng = np.random.default_rng(args.seed)
    index = rng.integers(0, args.D, size=args.N)
    index2 = rng.integers(0, args.D, size=args.N)
    dL = rng.normal(size=(args.N, args.N))

    def loops():
        out = np.zeros((args.D, args.D))
        for i in range(args.D):
            tmp1 = dL[index == i]
            for j in range(args.D):
                out[j, i] = tmp1[:, index2 == j].sum()
        return out

    def blas():
        return coregionalize_ops.gradient_reduce(dL, index, index2, args.D)

    impls = {"loops": loops, "blas": blas}
    if have_cython:
        idx = index.astype(np.int64)
        idx2 = index2.astype(np.int64)

        def cython():
            return coregionalize_cython.gradient_reduce(args.D, dL, idx, idx2)

        impls["cython"] = cython

    ref = loops() if args.N <= 2000 else blas()
    print("CIP-000E coregionalize gradient_reduce benchmark")
    for k, v in _env().items():
        print(f"  {k}: {v}")
    print(f"  N={args.N} D={args.D} reps={args.reps}")
    print()
    print(f"{'impl':<10} {'seconds':>10} {'×cython':>10} {'peak_KiB':>10} {'max_abs_err':>12}")
    rows = []
    for name, fn in impls.items():
        sec, peak, out = _time(fn, args.reps)
        err = float(np.max(np.abs(out - ref)))
        rows.append((name, sec, peak, err))
    cy = next((sec for name, sec, _, _ in rows if name == "cython"), None)
    for name, sec, peak, err in rows:
        rel = f"{sec / cy:10.3f}" if cy else f"{'—':>10}"
        print(f"{name:<10} {sec:10.4f} {rel} {peak:10.1f} {err:12.3e}")
    print()
    print("×cython < 1 means faster than Cython. blas = one-hot O.T @ dL @ O2 (GPy layout).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
