# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.core.model import Model
from .parameterization import Param, Parameterized
from . import parameterization

from .gp import GP
from .svgp import SVGP
from .sparse_gp import SparseGP
from .gp_grid import GpGrid
from .mapping import *


#===========================================================================
# Handle priors, this needs to be
# cleaned up at some point
#===========================================================================
def randomize(self, rand_gen=None, *args, **kwargs):
    """
    Randomize the model.
    Make this draw from the prior if one exists, else draw from given random generator

    :param rand_gen: np random number generator which takes args and kwargs
    :param flaot loc: loc parameter for random number generator
    :param float scale: scale parameter for random number generator
    :param args, kwargs: will be passed through to random number generator
    """
    if rand_gen is None:
        rand_gen = np.random.normal
    # first take care of all parameters (from N(0,1))
    x = rand_gen(size=self._size_transformed(), *args, **kwargs)
    updates = self.update_model()
    self.update_model(False) # Switch off the updates
    self.optimizer_array = x  # makes sure all of the tied parameters get the same init (since there's only one prior object...)
    # now draw from prior where possible
    x = self.param_array.copy()
    [np.put(x, ind, p.rvs(ind.size)) for p, ind in self.priors.items() if not p is None]
    unfixlist = np.ones((self.size,),dtype=np.bool)
    from paramz.transformations import __fixed__
    unfixlist[self.constraints[__fixed__]] = False
    self.param_array.flat[unfixlist] = x.view(np.ndarray).ravel()[unfixlist]
    self.update_model(updates)
    
Model.randomize = randomize
Param.randomize = randomize
Parameterized.randomize = randomize
