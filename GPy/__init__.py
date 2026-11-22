# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# Do not filter DeprecationWarning process-wide here (CIP-0010 / REQ-0003).
# A global ignore hid GPy's own deprecations and silenced third-party
# warnings for every process that imports GPy. User-facing GPy
# deprecations should use FutureWarning (visible by default); see
# Discussion #1297.

from . import core
from . import models
from . import mappings
from . import inference
from . import util
from . import examples
from . import likelihoods
from . import testing
from . import kern
from . import plotting

from .util import normalizer

# backwards compatibility
import sys

backwards_compatibility = ["lists_and_dicts", "observable_array", "index_operations"]
for bc in backwards_compatibility:
    sys.modules[f"GPy.core.parameterization.{bc!s}"] = getattr(core.parameterization, bc)

# CIP-000F Stage 3: real module for ObsAr aliases (Accept pickle policy).
# Previous-release pickles resolve ``paramz.core.observable_array.ObsAr`` while
# paramz remains installed; GPy path is registered for new references / Stage 4.
from .core.parameterization import observable_array as _gpy_observable_array

sys.modules["GPy.core.parameterization.observable_array"] = _gpy_observable_array

# Direct imports for convenience:
from .core import Model
from .core.parameterization import priors
from .core.parameterization import (
    Param,
    Parameterized,
    ObsAr,
    transformations as constraints,
)

from .__version__ import __version__


def load(file_or_path):
    """
    Load a previously pickled model, using `m.pickle('path/to/file.pickle)'

    :param file_name: path/to/file.pickle
    """
    # This is the pickling pain when changing _src -> src
    import sys
    import inspect

    sys.modules["GPy.kern._src"] = kern.src
    for name, module in inspect.getmembers(kern.src):
        if not name.startswith("_"):
            sys.modules[f"GPy.kern._src.{name}"] = module
    sys.modules["GPy.inference.optimization"] = inference.optimization
    from GPy.core.parameterization import paramz_load

    return paramz_load(file_or_path)
