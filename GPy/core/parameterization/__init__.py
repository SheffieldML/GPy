"""
Introduction
^^^^^^^^^^^^

Extends the functionality of the `paramz` package (dependency) to support paramterization of priors (:py:class:`GPy.core.parameterization.priors`).

.. inheritance-diagram:: GPy.core.parameterization.priors
   :top-classes: paramz.core.parameter_core.Parameterizable
"""


# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .param import Param
from .parameterized import Parameterized
from . import transformations

from paramz.core import lists_and_dicts, index_operations, observable_array, observable
from paramz import ObsAr
