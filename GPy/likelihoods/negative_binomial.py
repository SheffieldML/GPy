# Copyright (c) 2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


try:
    import sympy as sym
    sympy_available=True
    from sympy.utilities.lambdify import lambdify
    from GPy.util.symbolic import gammaln, logisticln
except ImportError:
    sympy_available=False

import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
import link_functions
from symbolic import Symbolic
from scipy import stats

if sympy_available:
    class Negative_binomial(Symbolic):
        """
        Negative binomial

        .. math::
            p(y_{i}|\pi(f_{i})) = \left(\frac{r}{r+f_i}\right)^r \frac{\Gamma(r+y_i)}{y!\Gamma(r)}\left(\frac{f_i}{r+f_i}\right)^{y_i}

        .. Note::
            Y takes non zero integer values..
            link function should have a positive domain, e.g. log (default).

        .. See also::
            symbolic.py, for the parent class
        """
        def __init__(self, gp_link=None):
            if gp_link is None:
                gp_link = link_functions.Identity()

            dispersion = sym.Symbol('dispersion', positive=True, real=True)
            y = sym.Symbol('y', nonnegative=True, integer=True)
            f = sym.Symbol('f', positive=True, real=True) 
            gp_link = link_functions.Log()
            log_pdf=dispersion*sym.log(dispersion) - (dispersion+y)*sym.log(dispersion+f) + gammaln(y+dispersion) - gammaln(y+1) - gammaln(dispersion) + y*sym.log(f)  
            #log_pdf= -(dispersion+y)*logisticln(f-sym.log(dispersion)) + gammaln(y+dispersion) - gammaln(y+1) - gammaln(dispersion) + y*(f-sym.log(dispersion))  
            super(Negative_binomial, self).__init__(log_pdf=log_pdf, gp_link=gp_link, name='Negative_binomial')

            # TODO: Check this.
            self.log_concave = False

