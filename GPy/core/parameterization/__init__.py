"""
Introduction
^^^^^^^^^^^^

Extends the functionality of the vendored paramz core (CIP-000F Stage 4) to
support parameterization of priors (:py:class:`GPy.core.parameterization.priors`).

.. inheritance-diagram:: GPy.core.parameterization.priors
   :top-classes: paramz.core.parameter_core.Parameterizable
"""


# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# CIP-000F Stage 4: install vendored paramz aliases before any ``import paramz``.
from . import _paramz_bootstrap  # noqa: F401

from .param import Param
from .parameterized import Parameterized
from . import transformations
from . import caching
from . import domains
from .caching import Cache_this, Cacher
from .domains import _NEGATIVE, _POSITIVE, _REAL

# CIP-000F Stage 0/3/4: symbols from vendored paramz (via aliases) + GPy ObsAr path.
from paramz.core import lists_and_dicts, index_operations, observable
from . import observable_array
from paramz.core.pickleable import Pickleable
from paramz.parameterized import ParametersChangedMeta
from .observable_array import ObsAr
from paramz import Model as ParamzModel
from paramz import load as paramz_load

__all__ = [
    "Param",
    "Parameterized",
    "transformations",
    "caching",
    "domains",
    "Cache_this",
    "Cacher",
    "_NEGATIVE",
    "_POSITIVE",
    "_REAL",
    "lists_and_dicts",
    "index_operations",
    "observable_array",
    "observable",
    "Pickleable",
    "ParametersChangedMeta",
    "ObsAr",
    "ParamzModel",
    "paramz_load",
]
