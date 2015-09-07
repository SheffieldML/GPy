# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

#Only works for scipy 0.12+
from scipy.special import erfcx, erf
import numpy as np

def ln_diff_erfs(x1, x2, return_sign=False):
    """Function for stably computing the log of difference of two erfs in a numerically stable manner.
    :param x1 : argument of the positive erf
    :type x1: ndarray
    :param x2 : argument of the negative erf
    :type x2: ndarray
    :return: tuple containing (log(abs(erf(x1) - erf(x2))), sign(erf(x1) - erf(x2)))

    Based on MATLAB code that was written by Antti Honkela and modified by David Luengo and originally derived from code by Neil Lawrence.
    """
    x1 = np.require(x1).real
    x2 = np.require(x2).real
    if x1.size==1:
        x1 = np.reshape(x1, (1, 1))
    if x2.size==1:
        x2 = np.reshape(x2, (1, 1))

    if x1.shape==x2.shape:
        v = np.zeros_like(x1)
    else:
        if x1.size==1:
            v = np.zeros(x2.shape)
        elif x2.size==1:
            v = np.zeros(x1.shape)
        else:
            raise ValueError("This function does not broadcast unless provided with a scalar.")

    if x1.size == 1:
        x1 = np.tile(x1, x2.shape)

    if x2.size == 1:
        x2 = np.tile(x2, x1.shape)

    sign = np.sign(x1 - x2)
    if x1.size == 1:
        if sign== -1:
            swap = x1
            x1 = x2
            x2 = swap
    else:
        I = sign == -1
        swap = x1[I]
        x1[I] = x2[I]
        x2[I] = swap

    with np.errstate(divide='ignore'):
        # switch off log of zero warnings.

        # Case 0: arguments of different sign, no problems with loss of accuracy
        I0 = np.logical_or(np.logical_and(x1>0, x2<0), np.logical_and(x2>0, x1<0)) # I1=(x1*x2)<0

        # Case 1: x1 = x2 so we have log of zero.
        I1 = (x1 == x2)

        # Case 2: Both arguments are non-negative
        I2 = np.logical_and(x1 > 0, np.logical_and(np.logical_not(I0),
                                                   np.logical_not(I1)))
        # Case 3: Both arguments are non-positive
        I3 = np.logical_and(np.logical_and(np.logical_not(I0),
                                           np.logical_not(I1)),
                            np.logical_not(I2))
        _x2 = x2.flatten()
        _x1 = x1.flatten()
        for group, flags in zip((0, 1, 2, 3), (I0, I1, I2, I3)):

            if np.any(flags):
                if not x1.size==1:
                    _x1 = x1[flags]
                if not x2.size==1:
                    _x2 = x2[flags]
                if group==0:
                    v[flags] = np.log( erf(_x1) - erf(_x2) )
                elif group==1:
                    v[flags] = -np.inf
                elif group==2:
                    v[flags] = np.log(erfcx(_x2)
                                   -erfcx(_x1)*np.exp(_x2**2
                                                      -_x1**2)) - _x2**2
                elif group==3:
                    v[flags] = np.log(erfcx(-_x1)
                                   -erfcx(-_x2)*np.exp(_x1**2
                                                          -_x2**2))-_x1**2

    # TODO: switch back on log of zero warnings.

    if return_sign:
        return v, sign
    else:
        if v.size==1:
            if sign==-1:
                v = v.view('complex64')
                v += np.pi*1j
        else:
            # Need to add in a complex part because argument is negative.
            v = v.view('complex64')
            v[I] += np.pi*1j

    return v
