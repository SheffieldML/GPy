# Copyright (c) 2012-2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""GPy-owned optimisation wrappers (CIP-000F Stage 1)."""

from . import stochastics
from . import verbose_optimization
from .optimization import (  # noqa: F401
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
)
from .verbose_optimization import VerboseOptimization  # noqa: F401

import sys

# Preserve legacy module aliases used by plotting / docs.
sys.modules["GPy.inference.optimization.stochastics"] = stochastics
sys.modules["GPy.inference.optimization.Optimizer"] = Optimizer

__all__ = [
    "Adam",
    "Opt_Adadelta",
    "Optimizer",
    "RProp",
    "SCG",
    "VerboseOptimization",
    "get_optimizer",
    "opt_SCG",
    "opt_bfgs",
    "opt_lbfgsb",
    "opt_simplex",
    "opt_tnc",
    "verbose_optimization",
    "stochastics",
]
