"""Optional Stage 2T pytest (skipped without jax). Not under GPy/testing CI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("jax")

_SPEC = importlib.util.spec_from_file_location(
    "exact_gp_rbf_lml_jax",
    Path(__file__).with_name("exact_gp_rbf_lml_jax.py"),
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)


def test_jax_lml_parity_and_transforms():
    r = _MOD.run_assessment()
    assert r["lml_abs_err"] < 1e-10
    assert abs(r["lml_jit"] - r["lml_jax"]) < 1e-10
    assert r["grad_abs_err"] < 1e-5
    assert r["predict_mean_finite"]
    assert r["predict_var_nonneg"]
