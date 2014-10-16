# Copyright (c) 2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import sympy as sym
from sympy.utilities.lambdify import lambdify
# does not exist! JH from GPy.util.symbolic import gammaln

import numpy as np
import link_functions
from symbolic import Symbolic
from scipy import stats

class SstudentT(Symbolic):
    """
    Symbolic variant of the Student-t distribution.

    .. math::

    .. Note::
        Y takes real values.
        link function is identity

    .. See also::
        symbolic.py, for the parent class
    """
    def __init__(self, gp_link=None, deg_free=5.0, t_scale2=1.0):
        parameters = {'deg_free':5.0, 't_scale2':1.0}
        if gp_link is None:
            gp_link = link_functions.Identity()

        # this likelihood has severe problems with likelihoods saturating ...
        y_0 = sym.Symbol('y_0', real=True)
        f_0 = sym.Symbol('f_0', real=True)
        nu = sym.Symbol('nu', positive=True, real=True)
        t_scale2 = sym.Symbol('t_scale2', positive=True, real=True)
        log_pdf = (gammaln((nu + 1) * 0.5)
                - gammaln(nu * 0.5)
                - 0.5*sym.log(t_scale2 * nu * sym.pi)
                   - 0.5*(nu + 1)*sym.log(1 + (1/nu)*(((y_0-f_0)**2)/t_scale2)))
        super(SstudentT, self).__init__(log_pdf=log_pdf, parameters=parameters, gp_link=gp_link, name='SstudentT')
        self.log_concave = False


            
