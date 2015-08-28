# Copyright (c) 2012, 2013 Ricardo Andrade
# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import ndtr as std_norm_cdf

#define a standard normal pdf
_sqrt_2pi = np.sqrt(2*np.pi)
def std_norm_pdf(x):
    x = np.clip(x,-1e150,1e150)
    return np.exp(-np.square(x)/2)/_sqrt_2pi

def inv_std_norm_cdf(x):
    """
    Inverse cumulative standard Gaussian distribution
    Based on Winitzki, S. (2008)
    """
    z = 2*x -1
    ln1z2 = np.log(1-z**2)
    a = 8*(np.pi -3)/(3*np.pi*(4-np.pi))
    b = 2/(np.pi * a) + ln1z2/2
    inv_erf = np.sign(z) * np.sqrt( np.sqrt(b**2 - ln1z2/a) - b )
    return np.sqrt(2) * inv_erf

