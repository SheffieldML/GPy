# Copyright (c) 2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import sympy as sym
#from GPy.util.symbolic import normcdfln
import numpy as np
import link_functions
from symbolic import Symbolic
from scipy import stats

class Skew_exponential(Symbolic):
    """
    Negative binomial

    .. math::

    .. Note::
        Y takes real values.
        link function is identity

    .. See also::
        symbolic.py, for the parent class
    """
    def __init__(self, gp_link=None, shape=1.0, scale=1.0):
        parameters={'scale':scale, 'shape':shape}
        if gp_link is None:
            gp_link = link_functions.Identity()

        #func_modules = [{'exp':clip_exp}]

        scale = sym.Symbol('scale', positive=True, real=True)
        shape = sym.Symbol('shape', real=True)
        y_0 = sym.Symbol('y_0', real=True)
        f_0 = sym.Symbol('f_0', real=True) 
        log_pdf=sym.log(shape)-sym.log(scale)-((y_0-f_0)/scale) + normcdfln(shape*(y_0-f_0)/scale) 
        super(Skew_exponential, self).__init__(log_pdf=log_pdf, gp_link=gp_link, name='Skew_exponential', parameters=parameters)

        # TODO: Check this.
        self.log_concave = True


            
