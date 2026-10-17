#!/usr/bin/env python3
"""Reproduce Discussion #958 / issue #1217: param_array save/load mismatch.

Bare::

    m.param_array[:] = loaded

does not notify paramz observers, so predictions stay at default hyperparameters
even though ``m.param_array`` prints the loaded values.

The README load recipe (``initialize=False`` + ``m[:] = loaded``) and a plain
``m[:] = loaded`` assignment both restore the original predictions.

Also reproduces on ``GPy.util.datasets.olympic_100m_men`` when a local dataset
cache is available (set ``GPY_DATASETS`` or rely on ``~/.config/GPy``).

Usage::

    PYTHONPATH=../paramz:. python scripts/repro_param_array_save_load.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

import GPy


def _mae(y, mu):
    return float(np.mean(np.abs(y - mu)))


def _report(label, y, mu_ref, model):
    mu, _ = model.predict(model.X)
    mae = _mae(y, mu)
    dmu = float(np.max(np.abs(mu_ref - mu)))
    ls = float(model.kern.lengthscale) if hasattr(model.kern, "lengthscale") else float("nan")
    print(
        f"  {label:40s}  MAE={mae:.6g}  max|dmu|={dmu:.3g}  "
        f"lengthscale={ls:.6g}  objective={float(model.objective_function()):.6g}"
    )
    return dmu


def synthetic_xy(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(40, 1)
    Y = np.sin(6 * X) + 0.1 * rng.randn(40, 1)
    return X, Y


def olympic_xy():
    from GPy.util import datasets as ds

    override = os.environ.get("GPY_DATASETS")
    if override:
        ds.data_path = override
    ds.overide_manual_authorize = True
    data = ds.olympic_100m_men()
    return data["X"], data["Y"]


def run_case(name, X, Y):
    print(f"\n=== {name}  X={X.shape} ===")
    m = GPy.models.GPRegression(X, Y, GPy.kern.RBF(X.shape[1]))
    m.optimize(messages=False, max_iters=200)
    params = m.param_array.copy()
    mu0, _ = m.predict(X)
    print(
        f"  {'optimized original':40s}  MAE={_mae(Y, mu0):.6g}  "
        f"lengthscale={float(m.rbf.lengthscale):.6g}  "
        f"objective={float(m.objective_function()):.6g}"
    )

    # README prose (broken): assign into param_array without notifying observers
    m_prose = GPy.models.GPRegression(X.copy(), Y.copy(), GPy.kern.RBF(X.shape[1]))
    m_prose.param_array[:] = params
    d_prose = _report("README prose: param_array[:] =", Y, mu0, m_prose)

    # Minimal fix: slice assignment on the model triggers updates
    m_slice = GPy.models.GPRegression(X.copy(), Y.copy(), GPy.kern.RBF(X.shape[1]))
    m_slice[:] = params
    d_slice = _report("fix: m[:] =", Y, mu0, m_slice)

    # Official README code block
    m_readme = GPy.models.GPRegression(X.copy(), Y.copy(), GPy.kern.RBF(X.shape[1]), initialize=False)
    m_readme.update_model(False)
    m_readme.initialize_parameter()
    m_readme[:] = params
    m_readme.update_model(True)
    d_readme = _report("README code block: initialize=False + m[:]", Y, mu0, m_readme)

    # Same buffer write, but force the missing notification
    m_trig = GPy.models.GPRegression(X.copy(), Y.copy(), GPy.kern.RBF(X.shape[1]))
    m_trig.param_array[:] = params
    m_trig._trigger_params_changed()
    d_trig = _report("param_array[:] = + _trigger_params_changed", Y, mu0, m_trig)

    ok = d_prose > 1e-6 and d_slice <= 1e-8 and d_readme <= 1e-8 and d_trig <= 1e-8
    print("  RESULT:", "LIVE BUG CONFIRMED" if ok else "unexpected outcome")
    return ok


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cases = []
    if not argv or "synthetic" in argv:
        cases.append(("synthetic",) + synthetic_xy())
    if "olympic" in argv:
        cases.append(("olympic_100m_men",) + olympic_xy())

    confirmed = True
    for name, X, Y in cases:
        confirmed = run_case(name, X, Y) and confirmed

    print("\nSummary: prose `param_array[:] =` fails to update views/predictions;")
    print("use `m[:] = loaded` (README code block) or trigger params-changed after assign.")
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
