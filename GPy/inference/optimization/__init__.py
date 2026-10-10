from . import stochastics

# Re-export paramz optimizers for GPy callers.
from paramz.optimization import (  # noqa: F401
    Adam,
    Opt_Adadelta,
    Optimizer,
    RProp,
    SCG,
    get_optimizer,
    opt_SCG,
    opt_bfgs,
    opt_lbfgsb,
    opt_simplex,
    opt_tnc,
    optimize,
    verbose_optimization,
)

import sys

sys.modules["GPy.inference.optimization.stochastics"] = stochastics
sys.modules["GPy.inference.optimization.Optimizer"] = Optimizer

__all__ = [
    "Adam",
    "Opt_Adadelta",
    "Optimizer",
    "RProp",
    "SCG",
    "get_optimizer",
    "opt_SCG",
    "opt_bfgs",
    "opt_lbfgsb",
    "opt_simplex",
    "opt_tnc",
    "optimize",
    "verbose_optimization",
    "stochastics",
]
