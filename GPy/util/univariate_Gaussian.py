# Copyright (c) 2012, 2013 Ricardo Andrade
# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
# The following std_norm_cdf is imported here to be available to other modules that import it
from scipy.special import ndtr as std_norm_cdf  # noqa: F401

# define a standard normal pdf
_sqrt_2pi = np.sqrt(2 * np.pi)


def std_norm_pdf(x):
    x = np.clip(x, -1e150, 1e150)
    return np.exp(-np.square(x) / 2) / _sqrt_2pi


def inv_std_norm_cdf(x):
    """
    Inverse cumulative standard Gaussian distribution
    Based on Winitzki, S. (2008)
    """
    z = 2 * x - 1
    ln1z2 = np.log(1 - z ** 2)
    a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
    b = 2 / (np.pi * a) + ln1z2 / 2
    inv_erf = np.sign(z) * np.sqrt(np.sqrt(b ** 2 - ln1z2 / a) - b)
    return np.sqrt(2) * inv_erf


def logPdfNormal(z):
    """
    Robust implementations of log pdf of a standard normal.

     @see [[https://github.com/mseeger/apbsint/blob/master/src/eptools/potentials/SpecfunServices.h original implementation]]
     in C from Matthias Seeger.
    """
    return -0.5 * (M_LN2PI + z * z)


def cdfNormal(z):
    """
    Robust implementations of cdf of a standard normal.

     @see [[https://github.com/mseeger/apbsint/blob/master/src/eptools/potentials/SpecfunServices.h original implementation]]
     in C from Matthias Seeger.
    */
    """
    if abs(z) < ERF_CODY_LIMIT1:
        # Phi(z) approx (1+y R_3(y^2))/2, y=z/sqrt(2)
        return 0.5 * (1.0 + (z / M_SQRT2) * _erfRationalHelperR3(0.5 * z * z))
    elif z < 0.0:
        # Phi(z) approx N(z)Q(-z)/(-z), z<0
        return np.exp(logPdfNormal(z)) * _erfRationalHelper(-z) / (-z)
    else:
        return 1.0 - np.exp(logPdfNormal(z)) * _erfRationalHelper(z) / z


def logCdfNormal(z):
    """
    Robust implementations of log cdf of a standard normal.

     @see [[https://github.com/mseeger/apbsint/blob/master/src/eptools/potentials/SpecfunServices.h original implementation]]
     in C from Matthias Seeger.
    """
    if abs(z) < ERF_CODY_LIMIT1:
        # Phi(z) approx  (1+y R_3(y^2))/2, y=z/sqrt(2)
        return np.log1p((z / M_SQRT2) * _erfRationalHelperR3(0.5 * z * z)) - M_LN2
    elif z < 0.0:
        # Phi(z) approx N(z)Q(-z)/(-z), z<0
        return logPdfNormal(z) - np.log(-z) + np.log(_erfRationalHelper(-z))
    else:
        return np.log1p(-(np.exp(logPdfNormal(z))) * _erfRationalHelper(z) / z)


def derivLogCdfNormal(z):
    """
    Robust implementations of derivative of the log cdf of a standard normal.

    @see [[https://github.com/mseeger/apbsint/blob/master/src/eptools/potentials/SpecfunServices.h original implementation]]
    in C from Matthias Seeger.
    """
    if abs(z) < ERF_CODY_LIMIT1:
        # Phi(z) approx (1 + y R_3(y^2))/2, y = z/sqrt(2)
        return (
            2.0
            * np.exp(logPdfNormal(z))
            / (1.0 + (z / M_SQRT2) * _erfRationalHelperR3(0.5 * z * z))
        )
    elif z < 0.0:
        # Phi(z) approx N(z) Q(-z)/(-z), z<0
        return -z / _erfRationalHelper(-z)
    else:
        t = np.exp(logPdfNormal(z))
        return t / (1.0 - t * _erfRationalHelper(z) / z)


def _erfRationalHelper(x):
    assert x > 0.0, "Arg of erfRationalHelper should be >0.0; was {}".format(x)

    if x >= ERF_CODY_LIMIT2:
        """
        x/sqrt(2) >= 4

        Q(x)   = 1 + sqrt(pi) y R_1(y),
        R_1(y) = poly(p_j,y) / poly(q_j,y),  where  y = 2/(x*x)

        Ordering of arrays: 4,3,2,1,0,5 (only for numerator p_j; q_5=1)
        ATTENTION: The p_j are negative of the entries here
        p (see P1_ERF)
        q (see Q1_ERF)
        """
        y = 2.0 / (x * x)

        res = y * P1_ERF[5]
        den = y
        i = 0

        while i <= 3:
            res = (res + P1_ERF[i]) * y
            den = (den + Q1_ERF[i]) * y
            i += 1

        # Minus, because p(j) values have to be negated
        return 1.0 - M_SQRTPI * y * (res + P1_ERF[4]) / (den + Q1_ERF[4])
    else:
        """
        x/sqrt(2) < 4, x/sqrt(2) >= 0.469

        Q(x)   = sqrt(pi) y R_2(y),
        R_2(y) = poly(p_j,y) / poly(q_j,y),   y = x/sqrt(2)

        Ordering of arrays: 7,6,5,4,3,2,1,0,8 (only p_8; q_8=1)
        p (see P2_ERF)
        q (see Q2_ERF
        """
        y = x / M_SQRT2
        res = y * P2_ERF[8]
        den = y
        i = 0

        while i <= 6:
            res = (res + P2_ERF[i]) * y
            den = (den + Q2_ERF[i]) * y
            i += 1

        return M_SQRTPI * y * (res + P2_ERF[7]) / (den + Q2_ERF[7])


def _erfRationalHelperR3(y):
    assert y >= 0.0, "Arg of erfRationalHelperR3 should be >=0.0; was {}".format(y)

    nom = y * P3_ERF[4]
    den = y
    i = 0
    while i <= 2:
        nom = (nom + P3_ERF[i]) * y
        den = (den + Q3_ERF[i]) * y
        i += 1
    return (nom + P3_ERF[3]) / (den + Q3_ERF[3])


ERF_CODY_LIMIT1 = 0.6629
ERF_CODY_LIMIT2 = 5.6569
M_LN2PI = 1.83787706640934533908193770913
M_LN2 = 0.69314718055994530941723212146
M_SQRTPI = 1.77245385090551602729816748334
M_SQRT2 = 1.41421356237309504880168872421

# weights for the erfHelpers (defined here to avoid redefinitions at every call)
P1_ERF = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
]
Q1_ERF = [
    2.56852019228982242e0,
    1.87295284992346047e0,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
]
P2_ERF = [
    5.64188496988670089e-1,
    8.88314979438837594e0,
    6.61191906371416295e1,
    2.98635138197400131e2,
    8.81952221241769090e2,
    1.71204761263407058e3,
    2.05107837782607147e3,
    1.23033935479799725e3,
    2.15311535474403846e-8,
]
Q2_ERF = [
    1.57449261107098347e1,
    1.17693950891312499e2,
    5.37181101862009858e2,
    1.62138957456669019e3,
    3.29079923573345963e3,
    4.36261909014324716e3,
    3.43936767414372164e3,
    1.23033935480374942e3,
]
P3_ERF = [
    3.16112374387056560e0,
    1.13864154151050156e2,
    3.77485237685302021e2,
    3.20937758913846947e3,
    1.85777706184603153e-1,
]
Q3_ERF = [
    2.36012909523441209e1,
    2.44024637934444173e2,
    1.28261652607737228e3,
    2.84423683343917062e3,
]
