# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

#Only works for scipy 0.12+
try:
    from scipy.special import erfcx, erf
except ImportError:
    from scipy.special import erf
    from erfcx import erfcx

import numpy as np

def ln_diff_erfs(x1, x2, return_sign=False):
    """Function for stably computing the log of difference of two erfs in a numerically stable manner.
    :param x1 : argument of the positive erf
    :type x1: ndarray
    :param x2 : argument of the negative erf
    :type x2: ndarray
    :return: tuple containing (log(abs(erf(x1) - erf(x2))), sign(erf(x1) - erf(x2)))
    
    Based on MATLAB code that was written by Antti Honkela and modified by David Luengo, originally derived from code by Neil Lawrence.
    """
    x1 = np.require(x1).real
    x2 = np.require(x2).real

    v = np.zeros(np.max((x1.size, x2.size)))
    
    # if numel(x1) == 1:
    #     x1 = x1 * ones(size(x2))

    # if numel(x2) == 1:
    #     x2 = x2 * ones(size(x1))

    sign = np.sign(x1 - x2)
    I = sign == -1
    swap = x1[I]
    x1[I] = x2[I]
    x2[I] = swap

    # TODO: switch off log of zero warnings.
    # Case 1: arguments of different sign, no problems with loss of accuracy
    I1 = np.logical_or(np.logical_and(x1>0, x2<0), np.logical_and(x2>0, x1<0)) # I1=(x1*x2)<0
    v[I1] = np.log( erf(x1[I1]) - erf(x2[I1]) )

    # Case 2: x1 = x2 so we have log of zero.
    I2 = (x1 == x2)
    v[I2] = -np.inf

    # Case 3: Both arguments are non-negative
    I3 = np.logical_and(x1 > 0, np.logical_and(np.logical_not(I1),
                                               np.logical_not(I2)))
    v[I3] = np.log(erfcx(x2[I3])
                   -erfcx(x1[I3])*np.exp(x2[I3]**2
                                         -x1[I3]**2)) - x2[I3]**2
    
    # Case 4: Both arguments are non-positive
    I4 = np.logical_and(np.logical_and(np.logical_not(I1),
                                       np.logical_not(I2)),
                        np.logical_not(I3))
    v[I4] = np.log(erfcx(-x1[I4])
                   -erfcx(-x2[I4])*np.exp(x1[I4]**2
                                          -x2[I4]**2))-x1[I4]**2
    
    # TODO: switch back on log of zero warnings.

    if return_sign:
        return v, sign
    else:
        # Need to add in a complex part because argument is negative.
        v[I] += np.pi*1j

