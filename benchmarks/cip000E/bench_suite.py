#!/usr/bin/env python3
"""CIP-000E: single-tree benchmark suite (public defaults + model-level).

Times the *wired* GPy entry points for this install — Cython on ``main``,
NumPy/BLAS on ``cip000E-numpy-blas-primitives`` — plus a few end-to-end
model steps that exercise those hotspots.

Emits JSON on stdout. Does not import JAX.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import tracemalloc
from typing import Any, Callable

import numpy as np


def _env() -> dict[str, Any]:
    import numpy
    import scipy

    try:
        blas = numpy.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    except TypeError:
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


def _time(fn: Callable[[], Any], reps: int) -> tuple[float, float, Any]:
    out = fn()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    elapsed = (time.perf_counter() - t0) / reps
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024.0, out


def _git_meta() -> dict[str, str]:
    import subprocess

    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "describe": _run(["git", "describe", "--always", "--dirty"]),
    }


def _probe_defaults() -> dict[str, Any]:
    from GPy.util import choleskies, linalg
    from GPy.kern.src import coregionalize, stationary

    def _name(obj: Any) -> str:
        return getattr(obj, "__name__", type(obj).__name__)

    return {
        "use_choleskies_cython": bool(getattr(choleskies, "use_choleskies_cython", False)),
        "use_linalg_cython": bool(getattr(linalg, "use_linalg_cython", False)),
        "use_stationary_cython": bool(getattr(stationary, "use_stationary_cython", False)),
        "use_coregionalize_cython": bool(getattr(coregionalize, "use_coregionalize_cython", False)),
        "backprop_gradient": _name(choleskies.backprop_gradient),
        "flat_to_triang": _name(choleskies.flat_to_triang),
        "triang_to_flat": _name(choleskies.triang_to_flat),
    }


def bench_cholesky_backprop(reps: int, seed: int) -> list[dict[str, Any]]:
    from GPy.util import choleskies

    rows = []
    for N in (500, 1000, 2000):
        rng = np.random.default_rng(seed + N)
        A = rng.normal(size=(N, N))
        K = A @ A.T + float(N) * np.eye(N)
        L = np.linalg.cholesky(K)
        dL = rng.normal(size=(N, N))
        sec, peak, _ = _time(lambda: choleskies.backprop_gradient(dL, L), reps)
        rows.append(
            {
                "bench": "cholesky_backprop",
                "N": N,
                "seconds": sec,
                "peak_kib": peak,
                "impl": _probe_defaults()["backprop_gradient"],
            }
        )
    return rows


def bench_triang_maps(reps: int, seed: int) -> list[dict[str, Any]]:
    from GPy.util import choleskies

    rows = []
    for M, D in ((50, 5), (200, 5), (400, 8)):
        Nflat = M * (M + 1) // 2
        rng = np.random.default_rng(seed + M)
        flat = rng.normal(size=(Nflat, D))
        sec_f, peak_f, L = _time(lambda: choleskies.flat_to_triang(flat), reps)
        sec_t, peak_t, _ = _time(lambda: choleskies.triang_to_flat(L), reps)
        rows.append(
            {
                "bench": "flat_to_triang",
                "M": M,
                "D": D,
                "seconds": sec_f,
                "peak_kib": peak_f,
                "impl": _probe_defaults()["flat_to_triang"],
            }
        )
        rows.append(
            {
                "bench": "triang_to_flat",
                "M": M,
                "D": D,
                "seconds": sec_t,
                "peak_kib": peak_t,
                "impl": _probe_defaults()["triang_to_flat"],
            }
        )
    return rows


def bench_symmetrify(reps: int, seed: int) -> list[dict[str, Any]]:
    from GPy.util import linalg

    rows = []
    for N in (500, 1000, 3000):
        rng = np.random.default_rng(seed + N)
        base = rng.normal(size=(N, N))

        def run():
            A = base.copy()
            linalg.symmetrify(A, upper=False)
            return A

        sec, peak, _ = _time(run, reps)
        rows.append(
            {
                "bench": "symmetrify",
                "N": N,
                "seconds": sec,
                "peak_kib": peak,
                "impl": "cython" if getattr(linalg, "use_linalg_cython", False) else "blocked_or_numpy",
            }
        )
    return rows


def bench_stationary(reps: int, seed: int) -> list[dict[str, Any]]:
    import GPy
    from GPy.kern.src import stationary as st_mod

    rows = []
    for N, Q in ((1000, 5), (3000, 10)):
        rng = np.random.default_rng(seed + N)
        k = GPy.kern.RBF(Q, ARD=True)
        X = rng.normal(size=(N, Q))
        X2 = rng.normal(size=(N // 2, Q))
        dK = rng.normal(size=(N, N // 2))
        # Lengthscale helpers take ``tmp = dL_dr * inv_dist`` (same as update_gradients_full).
        dL_dr = k.dK_dr_via_X(X, X2) * dK
        tmp = dL_dr * k._inv_dist(X, X2)

        sec_x, peak_x, _ = _time(lambda: k.gradients_X(dK, X, X2), reps)

        if getattr(st_mod, "use_stationary_cython", False) and hasattr(k, "_lengthscale_grads_cython"):
            len_fn = lambda: k._lengthscale_grads_cython(tmp, X, X2)
            len_impl = "cython"
        elif hasattr(k, "_lengthscale_grads_blas"):
            len_fn = lambda: k._lengthscale_grads_blas(tmp, X, X2)
            len_impl = "blas"
        else:
            len_fn = lambda: k._lengthscale_grads_pure(tmp, X, X2)
            len_impl = "pure"
        sec_l, peak_l, _ = _time(len_fn, reps)

        rows.append(
            {
                "bench": "gradients_X",
                "N": N,
                "M": N // 2,
                "Q": Q,
                "seconds": sec_x,
                "peak_kib": peak_x,
                "impl": "cython" if getattr(st_mod, "use_stationary_cython", False) else "blas",
            }
        )
        rows.append(
            {
                "bench": "lengthscale_grads",
                "N": N,
                "M": N // 2,
                "Q": Q,
                "seconds": sec_l,
                "peak_kib": peak_l,
                "impl": len_impl,
            }
        )
    return rows


def bench_coregionalize(reps: int, seed: int) -> list[dict[str, Any]]:
    import GPy
    from GPy.kern.src import coregionalize as cg_mod

    rows = []
    for N, D in ((2000, 5), (5000, 5)):
        rng = np.random.default_rng(seed + N)
        k = GPy.kern.Coregionalize(1, output_dim=D)
        index = rng.integers(0, D, size=(N, 1))
        index2 = rng.integers(0, D, size=(N, 1))
        dL = rng.normal(size=(N, N))
        if hasattr(k, "_gradient_reduce_blas") and not getattr(cg_mod, "use_coregionalize_cython", False):
            # Branch: BLAS is default via update path; call reduce helper directly.
            if hasattr(k, "_gradient_reduce_blas"):
                fn = lambda: k._gradient_reduce_blas(dL, index, index2)
                impl = "blas"
            else:
                fn = lambda: k.update_gradients_full(dL, index, index2)
                impl = "update_full"
        elif hasattr(k, "_gradient_reduce_cython") and getattr(cg_mod, "use_coregionalize_cython", False):
            fn = lambda: k._gradient_reduce_cython(dL, index, index2)
            impl = "cython"
        else:
            fn = lambda: k._gradient_reduce_numpy(dL, index, index2)
            impl = "numpy_loops"
        sec, peak, _ = _time(fn, reps)
        rows.append(
            {
                "bench": "coregionalize_reduce",
                "N": N,
                "D": D,
                "seconds": sec,
                "peak_kib": peak,
                "impl": impl,
            }
        )
        # K path (main Cython vs numpy indexing)
        X = index.astype(float)
        X2 = index2.astype(float)
        sec_k, peak_k, _ = _time(lambda: k.K(X, X2), reps)
        rows.append(
            {
                "bench": "coregionalize_K",
                "N": N,
                "D": D,
                "seconds": sec_k,
                "peak_kib": peak_k,
                "impl": "cython" if getattr(cg_mod, "use_coregionalize_cython", False) else "numpy",
            }
        )
    return rows


def bench_models(reps: int, seed: int) -> list[dict[str, Any]]:
    """End-to-end steps that hit Cholesky / stationary paths."""
    import GPy

    rows = []
    rng = np.random.default_rng(seed)

    # Dense GP: N=400, Q=3 — factorisation + gradients
    N, Q = 400, 3
    X = rng.normal(size=(N, Q))
    y = np.sin(X[:, :1]) + 0.1 * rng.normal(size=(N, 1))
    m = GPy.models.GPRegression(X, y, GPy.kern.RBF(Q, ARD=True))
    # Bust paramz cache each call so we time real factorisation/gradients.
    base_var = float(np.asarray(m.kern.variance).reshape(-1)[0])

    def gp_obj():
        m.kern.variance = base_var * (1.0 + 1e-9 * rng.random())
        return float(m.objective_function())

    def gp_grad():
        m.kern.variance = base_var * (1.0 + 1e-9 * rng.random())
        m.objective_function_gradients()
        return 1.0

    sec_o, peak_o, _ = _time(gp_obj, reps)
    sec_g, peak_g, _ = _time(gp_grad, reps)
    rows.append({"bench": "gp_objective", "N": N, "Q": Q, "seconds": sec_o, "peak_kib": peak_o, "impl": "GPRegression"})
    rows.append({"bench": "gp_gradients", "N": N, "Q": Q, "seconds": sec_g, "peak_kib": peak_g, "impl": "GPRegression"})

    # Sparse GP: N=2000, M=100 inducing — Cholesky of Kuu + stationary grads
    N, M_ind, Q = 2000, 100, 4
    X = rng.normal(size=(N, Q))
    y = np.sin(X[:, :1]) + 0.1 * rng.normal(size=(N, 1))
    Z = rng.normal(size=(M_ind, Q))
    m = GPy.models.SparseGPRegression(X, y, GPy.kern.RBF(Q, ARD=True), Z=Z)
    base_var = float(np.asarray(m.kern.variance).reshape(-1)[0])

    def sgp_obj():
        m.kern.variance = base_var * (1.0 + 1e-9 * rng.random())
        return float(m.objective_function())

    def sgp_grad():
        m.kern.variance = base_var * (1.0 + 1e-9 * rng.random())
        m.objective_function_gradients()
        return 1.0

    sec_o, peak_o, _ = _time(sgp_obj, max(1, reps // 2))
    sec_g, peak_g, _ = _time(sgp_grad, max(1, reps // 2))
    rows.append(
        {
            "bench": "sparse_gp_objective",
            "N": N,
            "M": M_ind,
            "Q": Q,
            "seconds": sec_o,
            "peak_kib": peak_o,
            "impl": "SparseGPRegression",
        }
    )
    rows.append(
        {
            "bench": "sparse_gp_gradients",
            "N": N,
            "M": M_ind,
            "Q": Q,
            "seconds": sec_g,
            "peak_kib": peak_g,
            "impl": "SparseGPRegression",
        }
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="Label for this tree (e.g. main, branch)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-models", action="store_true")
    args = p.parse_args(argv)

    import GPy

    rows: list[dict[str, Any]] = []
    rows.extend(bench_cholesky_backprop(args.reps, args.seed))
    rows.extend(bench_triang_maps(args.reps, args.seed))
    rows.extend(bench_symmetrify(args.reps, args.seed))
    rows.extend(bench_stationary(args.reps, args.seed))
    rows.extend(bench_coregionalize(args.reps, args.seed))
    if not args.skip_models:
        rows.extend(bench_models(args.reps, args.seed))

    payload = {
        "label": args.label,
        "gpy_version": getattr(GPy, "__version__", "unknown"),
        "git": _git_meta(),
        "defaults": _probe_defaults(),
        "env": _env(),
        "reps": args.reps,
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
