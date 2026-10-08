# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
Introduction
^^^^^^^^^^^^

This module contains the fundamental classes of GPy - classes that are
inherited by objects in other parts of GPy in order to provide a
consistent interface to major functionality.

.. inheritance-diagram:: GPy.core.gp.GP
    :top-classes: paramz.core.parameter_core.Parameterizable

:py:class:`GPy.core.model` is inherited by
:py:class:`GPy.core.gp.GP`. And :py:class:`GPy.core.model` itself
inherits :py:class:`paramz.model.Model` from the `paramz`
package. `paramz` essentially provides an inherited set of properties
and functions used to manage state (and state changes) of the model.

:py:class:`GPy.core.gp.GP` represents a GP model. Such an entity is
typically passed variables representing known (x) and observed (y)
data, along with a kernel and other information needed to create the
specific model. It exposes functions which return information derived
from the inputs to the model, for example predicting unobserved
variables based on new known variables, or the log marginal likelihood
of the current state of the model.

:py:func:`~GPy.core.gp.GP.optimize` is called to optimize
hyperparameters of the model. The optimizer argument takes a string
which is used to specify non-default optimization schemes.

Various plotting functions can be called against :py:class:`GPy.core.gp.GP`.

.. inheritance-diagram:: GPy.core.gp_grid.GpGrid GPy.core.sparse_gp.SparseGP GPy.core.sparse_gp_mpi.SparseGP_MPI GPy.core.svgp.SVGP 
    :top-classes: GPy.core.gp.GP

:py:class:`GPy.core.gp.GP` is used as the basis for classes supporting
more specialized types of Gaussian Process model. These are however
generally still not specific enough to be called by the user and are
inhereted by members of the :py:class:`GPy.models` package.

"""

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
    unfixlist = np.ones((self.size,),dtype=bool)
    from paramz.transformations import __fixed__
    unfixlist[self.constraints[__fixed__]] = False
    self.param_array.flat[unfixlist] = x.view(np.ndarray).ravel()[unfixlist]
    self.update_model(updates)
    
Model.randomize = randomize
Param.randomize = randomize
Parameterized.randomize = randomize
