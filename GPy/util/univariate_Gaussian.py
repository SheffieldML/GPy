# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import weave

def std_norm_pdf(x):
    """Standard Gaussian density function"""
    return 1./np.sqrt(2.*np.pi)*np.exp(-.5*x**2)

def std_norm_cdf(x):
    """
    Cumulative standard Gaussian distribution
    Based on Abramowitz, M. and Stegun, I. (1970)
    """
    support_code = "#include <math.h>"
    code = """

    double sign = 1.0;
    if (x < 0.0){
        sign = -1.0;
        x = -x;
    }
    x = x/sqrt(2.0);

    double t = 1.0/(1.0 +  0.3275911*x);

    double erf = 1. - exp(-x*x)*t*(0.254829592 + t*(-0.284496736 + t*(1.421413741 + t*(-1.453152027 + t*(1.061405429)))));

    return_val = 0.5*(1.0 + sign*erf);
    """
    x = float(x)
    return weave.inline(code,arg_names=['x'],support_code=support_code)


