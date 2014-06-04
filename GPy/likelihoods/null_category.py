# Copyright (c) 2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


try:
    import sympy as sym
    sympy_available=True
    from sympy.utilities.lambdify import lambdify
    from GPy.util.symbolic import gammaln, normcdfln, normcdf
    from sympy.functions.elementary.piecewise import Piecewise
except ImportError:
    sympy_available=False

import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
import link_functions
from symbolic import Symbolic
from scipy import stats

if sympy_available:
    class Null_category(Symbolic):
        """
        Null category noise model.

        .. math::

        .. Note::
            Y takes -1, 0 or 1.

        .. See also::
            symbolic.py, for the parent class
        """
        def __init__(self, gp_link=None):
            if gp_link is None:
                gp_link = link_functions.Identity()
            # width of the null category.
            width = sym.Symbol('width', positive=True, real=True)
            # prior probability of positive class
            p = sym.Symbol('p', positive=True, real=True)
            y = sym.Symbol('y', binary=True)            
            f = sym.Symbol('f', positive=True, real=True) 

            log_pdf_missing = sym.log((1-p)*normcdf(-f-width/2)
                                      +p*normcdf(f+width/2))
            log_pdf = (y-1)*normcdfln(-f-width/2)+y*normcdfln(f+width/2)
            super(Null_category, self).__init__(log_pdf=log_pdf, missing_log_pdf=log_pdf_missing, gp_link=gp_link, name='Null_category')

            self.p=0.5
            self.p.constrain_bounded(0., 1.)
            self.width = 1.
            self.width.constrain_fixed()
            self.log_concave = False

