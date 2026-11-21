#!/usr/bin/env python3
"""CIP-000E: stationary lengthscale_grads / gradients_X harness.

Compares pure (loop), blas (CIP-cited matmul/GEMM), and cython (if built).
"""

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
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--M", type=int, default=None, help="defaults to N")
    p.add_argument("--Q", type=int, default=10)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    M = args.N if args.M is None else args.M

    from GPy.kern.src import stationary_ops

    try:
        from GPy.kern.src import stationary_cython

        have_cython = True
    except ImportError:
        have_cython = False

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.N, args.Q))
    X2 = rng.normal(size=(M, args.Q))
    tmp = rng.normal(size=(args.N, M))

    def pure_len():
        return np.array(
            [np.sum(tmp * np.square(X[:, q : q + 1] - X2[:, q : q + 1].T)) for q in range(args.Q)]
        )

    def blas_len():
        return stationary_ops.lengthscale_grads_sqdiff(tmp, X, X2)

    def pure_gradX():
        grad = np.empty(X.shape)
        for q in range(args.Q):
            np.sum(tmp * (X[:, q][:, None] - X2[:, q][None, :]), axis=1, out=grad[:, q])
        return grad

    def blas_gradX():
        return stationary_ops.gradients_X_from_tmp(tmp, X, X2)

    impls = {
        "len_pure": pure_len,
        "len_blas": blas_len,
        "gradX_pure": pure_gradX,
        "gradX_blas": blas_gradX,
    }
    if have_cython:
        Xc, X2c = np.ascontiguousarray(X), np.ascontiguousarray(X2)

        def cy_len():
            g = np.zeros(args.Q)
            stationary_cython.lengthscale_grads(args.N, M, args.Q, tmp, X, X2, g)
            return g

        def cy_gradX():
            g = np.zeros(X.shape)
            stationary_cython.grad_X(args.N, args.Q, M, Xc, X2c, tmp, g)
            return g

        impls["len_cython"] = cy_len
        impls["gradX_cython"] = cy_gradX

    ref_len = pure_len() if args.N <= 800 else blas_len()
    ref_grad = pure_gradX() if args.N <= 800 else blas_gradX()

    print("CIP-000E stationary grads benchmark")
    for k, v in _env().items():
        print(f"  {k}: {v}")
    print(f"  N={args.N} M={M} Q={args.Q} reps={args.reps}")
    print()
    print(f"{'impl':<14} {'seconds':>10} {'×cython':>10} {'peak_KiB':>10} {'max_abs_err':>12}")

    timed = {}
    for name, fn in impls.items():
        if name.endswith("_pure") and args.N > 800:
            continue
        sec, peak, out = _time(fn, args.reps)
        timed[name] = sec
        if name.startswith("len_"):
            err = float(np.max(np.abs(out - ref_len)))
        else:
            err = float(np.max(np.abs(out - ref_grad)))
        timed[name + "_err"] = err
        timed[name + "_peak"] = peak

    for kind in ("len", "gradX"):
        cy = timed.get(f"{kind}_cython")
        for impl in ("pure", "blas", "cython"):
            key = f"{kind}_{impl}"
            if key not in timed:
                continue
            rel = timed[key] / cy if cy else float("nan")
            rel_s = f"{rel:10.3f}" if cy else f"{'—':>10}"
            print(
                f"{key:<14} {timed[key]:10.4f} {rel_s} "
                f"{timed[key + '_peak']:10.1f} {timed[key + '_err']:12.3e}"
            )
    print()
    print("×cython < 1 means faster than Cython. blas = CIP-cited matmul/GEMM forms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
