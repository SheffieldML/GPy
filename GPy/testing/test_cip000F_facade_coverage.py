"""Patch-coverage probes for CIP-000F Stage 0/1 façade and optimise dispatch.

Codecov only counts ``pytest GPy/testing``. Stage 0 left seven patch lines
cold (``GPy.load``, VarDTC pickle restore, ss_gplvm / ss_mrd transformation
imports). Stage 1 adds GPy-owned ``Model.optimize`` / SCG paths that need the
same treatment.
"""

from __future__ import annotations

import sys
import tempfile
import types
from unittest import mock

import numpy as np
import pytest

import GPy
from GPy.core import parameterization as gpy_param
from GPy.inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch
from GPy.inference.optimization import (
    Adam,
    Opt_Adadelta,
    Optimizer,
    RProp,
    SCG,
    VerboseOptimization,
    get_optimizer,
    opt_SCG,
    opt_bfgs,
    opt_lbfgsb,
    opt_simplex,
    opt_tnc,
)
from GPy.inference.optimization import optimization as opt_mod
from GPy.inference.optimization import verbose_optimization as verb_mod
from GPy.models.ss_gplvm import IBPPosterior, IBPPrior, SLVMPosterior
from GPy.models.ss_mrd import IBPPrior_SSMRD


def _toy_gp():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 1)
    Y = np.sin(X) + 0.05 * rng.randn(12, 1)
    return GPy.models.GPRegression(X, Y)


def _quad():
    """Simple quadratic for direct optimiser smoke tests."""

    def f(x):
        return float(np.sum(x**2))

    def fp(x):
        return 2.0 * x

    def f_fp(x):
        return f(x), fp(x)

    return f, fp, f_fp


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
        assert m.optimization_runs[-1].__class__.__module__.startswith("GPy.inference.optimization")

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
        m = _toy_gp()
        for name in ("tnc", "simplex", "bfgs"):
            m.optimize(optimizer=name, max_iters=3, messages=False)
        text = str(m.optimization_runs[-1])
        assert "Optimizer:" in text
        with pytest.raises(KeyError):
            get_optimizer("not-a-real-optimizer")


class TestStage1OptimizerBranches:
    def test_domains_constants_match_legacy_strings(self):
        assert gpy_param.domains._REAL == "real"
        assert gpy_param.domains._POSITIVE == "positive"
        assert gpy_param.domains._NEGATIVE == "negative"
        assert gpy_param.domains._BOUNDED == "bounded"

    def test_optimizer_base_opt_and_getstate(self):
        base = Optimizer(max_iters=2)
        with pytest.raises(NotImplementedError):
            base.opt(np.zeros(2))
        assert base.__getstate__() is base.__dict__

    def test_get_optimizer_aliases(self):
        assert get_optimizer("fmin_tnc") is opt_tnc
        assert get_optimizer("org-bfgs") is opt_bfgs
        assert get_optimizer("lbfgsb") is opt_lbfgsb
        assert get_optimizer("scg") is opt_SCG
        assert get_optimizer("adadelta") is Opt_Adadelta
        assert get_optimizer("rprop") is RProp
        assert get_optimizer("adam") is Adam

    def test_scipy_optimizers_with_tolerances(self):
        f, fp, f_fp = _quad()
        x0 = np.array([0.5, -0.25])
        tnc = opt_tnc(max_iters=20, xtol=1e-6, ftol=1e-6, gtol=1e-6)
        tnc.run(x0.copy(), f_fp=f_fp)
        assert tnc.x_opt is not None
        assert tnc.status is not None

        lbfgs = opt_lbfgsb(max_iters=20, xtol=1e-6, ftol=1e-6, gtol=1e-5, bfgs_factor=10.0)
        lbfgs.run(x0.copy(), f_fp=f_fp)
        assert lbfgs.x_opt is not None

        bfgs = opt_bfgs(max_iters=20, xtol=1e-6, ftol=1e-6, gtol=1e-5)
        bfgs.run(x0.copy(), f=f, fp=fp, f_fp=f_fp)
        assert bfgs.x_opt is not None

        simplex = opt_simplex(max_iters=30, xtol=1e-6, ftol=1e-6, gtol=1e-5)
        simplex.run(x0.copy(), f=f)
        assert simplex.x_opt is not None

    def test_scg_deprecation_and_direct_scg(self):
        f, fp, _f_fp = _quad()
        with pytest.warns(FutureWarning, match="max_f_eval deprecated"):
            opt = opt_SCG(max_iters=15, max_f_eval=100)
        opt.run(np.array([0.3, -0.1]), f=f, fp=fp)
        assert opt.x_opt is not None
        assert opt.trace is not None

        x, flog, n_eval, status = SCG(f, fp, np.array([0.2, 0.1]), maxiters=10)
        assert x.shape == (2,)
        assert len(flog) >= 1
        assert n_eval >= 1
        assert isinstance(status, str)

    def test_climin_constructors_require_dependency(self):
        with pytest.raises(ImportError, match="climin"):
            Opt_Adadelta(max_iters=2)
        with pytest.raises(ImportError, match="climin"):
            RProp(max_iters=2)
        with pytest.raises(ImportError, match="climin"):
            Adam(max_iters=2)
        with pytest.raises(ImportError, match="climin"):
            opt_mod._check_for_climin()

    def test_climin_optimizers_with_fake_backend(self):
        class _FakeOpt:
            def __init__(self, x, *args, **kwargs):
                self.wrt = np.asarray(x, dtype=float).copy()
                self._n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self._n += 1
                if self._n > 3:
                    raise StopIteration
                self.wrt *= 0.5
                return {"n_iter": self._n}

        fake_climin = types.ModuleType("climin")
        fake_climin.adadelta = types.SimpleNamespace(Adadelta=_FakeOpt)
        fake_climin.rprop = types.SimpleNamespace(Rprop=_FakeOpt)
        fake_climin.adam = types.SimpleNamespace(Adam=_FakeOpt)

        f, fp, _ = _quad()
        x0 = np.array([1.0, -1.0])
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "climin": fake_climin,
                    "climin.adadelta": fake_climin.adadelta,
                    "climin.rprop": fake_climin.rprop,
                    "climin.adam": fake_climin.adam,
                },
            ),
            mock.patch.object(opt_mod, "_check_for_climin", lambda: None),
        ):
            for cls in (Opt_Adadelta, RProp, Adam):
                opt = cls(max_iters=2)
                opt.run(x0.copy(), fp=fp)
                assert opt.x_opt is not None
                assert "maximum" in opt.status.lower()

    def test_verbose_console_path_and_time_formats(self):
        m = _toy_gp()
        m.optimize(optimizer="scg", max_iters=4, messages=True, ipython_notebook=False)
        assert m.optimization_runs[-1].opt_name

        vo = VerboseOptimization(
            m,
            m.optimization_runs[-1],
            maxiters=4,
            verbose=True,
            ipython_notebook=False,
        )
        vo.start = 0.0
        vo._time = 0.0
        for seconds in (0.5, 65.0, 3700.0, 90000.0):
            vo.print_out(seconds)
            assert vo.timestring
        vo.update()
        assert np.isfinite(vo.fnow)
        # Force the nan-gradient branch in VerboseOptimization.update.
        stub = types.SimpleNamespace(objective_function=lambda: 1.0, obj_grads=None)
        vo.model = stub
        vo.update()
        assert np.isnan(vo.current_gradient)
        assert verb_mod.exponents(1.0, np.nan).shape == (2,)

    def test_optimize_restarts_robust_and_all_fail_reset(self):
        m = _toy_gp()
        before = m.optimizer_array.copy()
        initial_len = len(m.optimization_runs)

        def boom(*_a, **_k):
            raise RuntimeError("forced fail")

        with mock.patch.object(m, "optimize", side_effect=boom):
            m.optimize_restarts(num_restarts=2, max_iters=2, verbose=True, robust=True)
        assert len(m.optimization_runs) == initial_len
        np.testing.assert_allclose(m.optimizer_array, before)

        with (
            mock.patch.object(m, "optimize", side_effect=boom),
            pytest.raises(RuntimeError, match="forced fail"),
        ):
            m.optimize_restarts(num_restarts=1, max_iters=2, verbose=False, robust=False)
