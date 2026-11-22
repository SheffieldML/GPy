#!/usr/bin/env python
"""CIP-000F Stage 3 wall-clock benchmarks (GPRegression / SparseGP / BGPLVM).

Run from repo root with the project venv::

    .venv/bin/python cip/cip000F/experiments/stage3_cache_benchmarks.py

Records mean wall-clock for log_likelihood + predict after a small optimize.
Not installed with the GPy package.
"""

from __future__ import annotations

import time

import numpy as np


def _time(fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def bench_gp_regression(n=80, d=2, repeats=5):
    import GPy

    np.random.seed(0)
    X = np.random.randn(n, d)
    Y = np.sin(X[:, :1]) + 0.1 * np.random.randn(n, 1)
    m = GPy.models.GPRegression(X, Y)
    m.optimize(max_iters=10)

    def work():
        m.log_likelihood()
        m.predict(X[:20])

    mean, std = _time(work, repeats=repeats)
    return {"model": "GPRegression", "n": n, "mean_s": mean, "std_s": std}


def bench_sparse_gp(n=200, d=2, m_ind=20, repeats=5):
    import GPy

    np.random.seed(1)
    X = np.random.randn(n, d)
    Y = np.sin(X[:, :1]) + 0.1 * np.random.randn(n, 1)
    m = GPy.models.SparseGPRegression(X, Y, num_inducing=m_ind)
    m.optimize(max_iters=10)

    def work():
        m.log_likelihood()
        m.predict(X[:30])

    mean, std = _time(work, repeats=repeats)
    return {
        "model": "SparseGPRegression",
        "n": n,
        "inducing": m_ind,
        "mean_s": mean,
        "std_s": std,
    }


def bench_bgplvm(n=60, d=5, q=2, repeats=3):
    import GPy

    np.random.seed(2)
    Y = np.random.randn(n, d)
    m = GPy.models.BayesianGPLVM(Y, input_dim=q, num_inducing=15)
    m.optimize(max_iters=5)

    def work():
        m.log_likelihood()
        m.predict(m.X.mean[:10])

    mean, std = _time(work, repeats=repeats)
    return {"model": "BayesianGPLVM", "n": n, "mean_s": mean, "std_s": std}


def main():
    rows = [
        bench_gp_regression(),
        bench_sparse_gp(),
        bench_bgplvm(),
    ]
    print("CIP-000F Stage 3 cache/ObsAr benchmarks (post-change)")
    print("{:<22} {:>8} {:>12} {:>10}".format("model", "n", "mean_s", "std_s"))
    for r in rows:
        print(
            "{:<22} {:>8} {:>12.4f} {:>10.4f}".format(
                r["model"], r["n"], r["mean_s"], r["std_s"]
            )
        )
    return rows


if __name__ == "__main__":
    main()
