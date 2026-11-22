# Copyright (c) 2014, Max Zwiessele, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from paramz import Parameterized
from .priorizable import Priorizable

import logging

logger = logging.getLogger("parameters changed meta")


class Parameterized(Parameterized, Priorizable):
    """
    Parameterized class

    CIP-000F Stage 3: ``_cache_version`` is bumped when parameters change so
    :class:`~GPy.core.parameterization.caching.Cache_this` entries miss without
    requiring ``ObsAr`` observers on data arrays.

    Say m is a handle to a parameterized class.

    Printing parameters:

        - print m:           prints a nice summary over all parameters
        - print m.name:      prints details for param with name 'name'
        - print m[regexp]: prints details for all the parameters
                             which match (!) regexp
        - print m['']:       prints details for all parameters

        Fields:

            Name:       The name of the param, can be renamed!
            Value:      Shape or value, if one-valued
            Constrain:  constraint of the param, curly "{c}" brackets indicate
                        some parameters are constrained by c. See detailed print
                        to get exact constraints.
            Tied_to:    which paramter it is tied to.

    Getting and setting parameters:

        Set all values in param to one:

            m.name.to.param = 1

    Handling of constraining, fixing and tieing parameters:

        You can constrain parameters by calling the constrain on the param itself, e.g:

            - m.name[:,1].constrain_positive()
            - m.name[0].tie_to(m.name[1])

        Fixing parameters will fix them to the value they are right now. If you change
        the parameters value, the param will be fixed to the new value!

        If you want to operate on all parameters use m[''] to wildcard select all paramters
        and concatenate them. Printing m[''] will result in printing of all parameters in detail.
    """

    def bump_cache_version(self):
        """Invalidate ``Cache_this`` entries keyed on this object's version."""
        self._cache_version = getattr(self, "_cache_version", 0) + 1
        cache = getattr(self, "cache", None)
        if cache is not None and hasattr(cache, "reset"):
            cache.reset()

    def parameters_changed(self):
        """Default hook: bump cache version. Subclasses may override without ``super``."""
        self.bump_cache_version()

    def __getstate__(self):
        state = super(Parameterized, self).__getstate__()
        # Compatible with paramz < 0.10.1: omit the parent link so pickling a
        # used subtree (e.g. a kernel attached to a GP) does not serialize the
        # whole model and hit incomplete-object cycles on load (#605, #932).
        # paramz >= 0.10.1 already omits `_parent_` in Pickleable.__getstate__.
        state.pop("_parent_", None)
        return state
