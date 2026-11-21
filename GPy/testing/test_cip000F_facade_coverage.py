"""Patch-coverage probes for CIP-000F Stage 0/1 façade and optimise dispatch.

Codecov only counts ``pytest GPy/testing``. Stage 0 left seven patch lines
cold (``GPy.load``, VarDTC pickle restore, ss_gplvm / ss_mrd transformation
imports). Stage 1 adds GPy-owned ``Model.optimize`` / SCG paths that need the
same treatment.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

import GPy
from GPy.inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
from GPy.models.ss_gplvm import IBPPosterior, IBPPrior, SLVMPosterior
from GPy.models.ss_mrd import IBPPrior_SSMRD


def _toy_gp():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 1)
    Y = np.sin(X) + 0.05 * rng.randn(12, 1)
    return GPy.models.GPRegression(X, Y)


class TestStage0FacadeCoverage:
    def test_gpy_load_uses_paramz_load_facade(self):
        m = _toy_gp()
        with tempfile.NamedTemporaryFile(suffix=".pickle") as f:
            m.pickle(f)
            f.flush()
            f.seek(0)
            m2 = GPy.load(f)
        np.testing.assert_allclose(m.param_array, m2.param_array)

    def test_vardtc_minibatch_setstate_rebuilds_cacher(self):
        inf = VarDTC_minibatch(batchsize=4, limit=2)
        state = inf.__getstate__()
        restored = VarDTC_minibatch.__new__(VarDTC_minibatch)
        restored.__setstate__(state)
        assert restored.batchsize == 4
        assert restored.limit == 2
        assert restored.get_trYYT.limit == 2

    def test_ss_gplvm_priors_import_gpy_transformations(self):
        means = np.zeros((5, 2))
        variances = np.ones((5, 2))
        binary = np.full((5, 2), 0.5)
        post = IBPPosterior(means, variances, binary)
        assert post.tau.shape == (2, 2)
        prior = IBPPrior(input_dim=2, alpha=1.5)
        np.testing.assert_allclose(np.asarray(prior.alpha).ravel(), [1.5])
        slvm = SLVMPosterior(means, variances, binary)
        assert slvm.tau.shape == (2, 2)

    def test_ss_mrd_prior_imports_gpy_fixed(self):
        prior = IBPPrior_SSMRD(nModels=2, input_dim=3, alpha=2.0)
        assert prior.nModels == 2
        assert prior.input_dim == 3


class TestStage1OptimizeCoverage:
    def test_optimize_scg_and_optimizer_instance(self):
        m = _toy_gp()
        m.optimize(optimizer="scg", max_iters=5, messages=False)
        assert m.optimization_runs
        assert m.optimization_runs[-1].__class__.__module__.startswith(
            "GPy.inference.optimization"
        )

        from GPy.inference.optimization import opt_lbfgsb

        opt = opt_lbfgsb(max_iters=5)
        m.optimize(optimizer=opt, max_iters=5, messages=False)
        assert m.optimization_runs[-1] is opt

    def test_optimize_nothing_to_optimize_when_fixed(self):
        m = _toy_gp()
        m[:] = m.param_array  # ensure values present
        m.constrain_fixed()
        assert m.optimize(max_iters=2) is None

    def test_optimize_restarts_and_updates_off(self):
        m = _toy_gp()
        m.update_model(False)
        m.optimize(max_iters=3, messages=False)
        assert m.update_model()

        m.optimize_restarts(num_restarts=2, max_iters=3, verbose=False, robust=True)
        assert len(m.optimization_runs) >= 2

    def test_opt_wrapper_roundtrip(self):
        from GPy.core.model import _opt_wrapper

        m = _toy_gp()
        opt = _opt_wrapper((m, {"max_iters": 2, "messages": False}))
        assert opt is m.optimization_runs[-1]

    def test_alternate_optimizers_and_diagnostics(self):
        from GPy.inference.optimization import get_optimizer

        m = _toy_gp()
        for name in ("tnc", "simplex", "bfgs"):
            m.optimize(optimizer=name, max_iters=3, messages=False)
        text = str(m.optimization_runs[-1])
        assert "Optimizer:" in text
        with pytest.raises(KeyError):
            get_optimizer("not-a-real-optimizer")
