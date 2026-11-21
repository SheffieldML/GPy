# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""CIP-000F Stage 4: register vendored paramz under the historical ``paramz`` names.

Import this module before any ``import paramz`` so pickle paths and the
parameterization façade resolve to :mod:`GPy.core.parameterization._paramz`.
Always prefers the in-tree vendor over a site-packages ``paramz``.
"""

from __future__ import annotations

import sys
from types import ModuleType


def install_paramz_aliases() -> ModuleType:
    """Place vendored paramz on ``sys.modules`` under the ``paramz`` names."""
    from . import _paramz as vendor

    sys.modules["paramz"] = vendor
    _alias_submodules(vendor)
    return vendor


def _alias_submodules(vendor: ModuleType) -> None:
    mapping = {
        "paramz.caching": "caching",
        "paramz.domains": "domains",
        "paramz.transformations": "transformations",
        "paramz.parameterized": "parameterized",
        "paramz.param": "param",
        "paramz.model": "model",
        "paramz.util": "util",
        "paramz.optimization": "optimization",
        "paramz.core": "core",
    }
    for mod_name, attr in mapping.items():
        try:
            sub = __import__(
                "GPy.core.parameterization._paramz." + attr,
                fromlist=["*"],
            )
        except ImportError:
            sub = getattr(vendor, attr, None)
            if sub is None:
                continue
        sys.modules[mod_name] = sub

    core = sys.modules.get("paramz.core")
    if core is None:
        return
    for leaf in (
        "observable",
        "observable_array",
        "lists_and_dicts",
        "index_operations",
        "pickleable",
        "parameter_core",
        "constrainable",
        "gradcheckable",
        "indexable",
        "nameable",
        "parentable",
        "updateable",
    ):
        key = "paramz.core." + leaf
        try:
            sys.modules[key] = __import__(
                "GPy.core.parameterization._paramz.core." + leaf,
                fromlist=["*"],
            )
        except ImportError:
            if hasattr(core, leaf):
                sys.modules[key] = getattr(core, leaf)


install_paramz_aliases()
