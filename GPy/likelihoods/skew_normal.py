# Copyright (c) 2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


try:
    import sympy as sym
    sympy_available=True
    from sympy.utilities.lambdify import lambdify
    from GPy.util.symbolic import normcdfln, normcdf
except ImportError:
    sympy_available=False

import numpy as np
from GPy.util.functions import clip_exp
import link_functions
from symbolic import Symbolic
from scipy import stats

class Skew_normal(Symbolic):
    """
    Skew Normal distribution.

    .. math::

    .. Note::
    Y takes real values.
    link function is identity

    .. See also::
    symbolic.py, for the parent class
    """
    def __init__(self, gp_link=None, shape=1.0, scale=1.0):
        parameters = {'scale': scale, 'shape':shape}
        if gp_link is None:
            gp_link = link_functions.Identity()

        # # this likelihood has severe problems with likelihoods saturating exponentials, so clip_exp is used in place of the true exp as a solution for dealing with the numerics.
        # func_modules = [{'exp':clip_exp}]
        func_modules = []

        scale = sym.Symbol('scale', positive=True, real=True)
        shape = sym.Symbol('shape', real=True)
        y_0 = sym.Symbol('y_0', real=True)
        f_0 = sym.Symbol('f_0', real=True) 
        log_pdf=-sym.log(scale)-1./2*sym.log(2*sym.pi)-1./2*((y_0-f_0)/scale)**2 + sym.log(2) + normcdfln(shape*(y_0-f_0)/scale) 
        super(Skew_normal, self).__init__(log_pdf=log_pdf, parameters=parameters, gp_link=gp_link, name='Skew_normal', func_modules=func_modules)

        self.log_concave = True
        


            
