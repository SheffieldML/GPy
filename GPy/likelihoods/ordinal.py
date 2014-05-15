# Copyright (c) 2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import sympy as sym
from GPy.util.symbolic import gammaln, normcdfln, normcdf, IndMatrix, create_matrix
import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
import link_functions
from symbolic import Symbolic
from scipy import stats

class Ordinal(Symbolic):
    """
    Ordinal

    .. math::
        p(y_{i}|\pi(f_{i})) = \left(\frac{r}{r+f_i}\right)^r \frac{\Gamma(r+y_i)}{y!\Gamma(r)}\left(\frac{f_i}{r+f_i}\right)^{y_i}

    .. Note::
        Y takes non zero integer values..
        link function should have a positive domain, e.g. log (default).

    .. See also::
        symbolic.py, for the parent class
    """
    def __init__(self, categories=3, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        dispersion = sym.Symbol('width', positive=True, real=True)
        y_0 = sym.Symbol('y_0', nonnegative=True, integer=True)
        f_0 = sym.Symbol('f_0', positive=True, real=True) 
        log_pdf = create_matrix('log_pdf', 1, categories)
        log_pdf[0] = normcdfln(-f_0)
        if categories>2:
            w = create_matrix('w', 1, categories)
            log_pdf[categories-1] = normcdfln(w.sum() + f_0)
            for i in range(1, categories-1):
                log_pdf[i] = sym.log(normcdf(w[0, 0:i-1].sum() + f_0) - normcdf(w[0, 0:i].sum()-f_0) )
        else:
            log_pdf[1] = normcdfln(f_0)
        log_pdf.index_var = y_0
        super(Ordinal, self).__init__(log_pdf=log_pdf, gp_link=gp_link, name='Ordinal')

        # TODO: Check this.
        self.log_concave = True

