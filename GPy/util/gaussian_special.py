# Copyright (c) 2026 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Gaussian special functions shared by likelihoods and LFM / SIM kernels.

Inverse Mills ratio
-------------------
``inv_mills_ratio(z) = φ(z)/Φ(z)`` is the derivative of ``log Φ(z)``. It appears
in probit / skew-normal likelihoods. The scalar Cody rational helpers in
``univariate_Gaussian.derivLogCdfNormal`` compute the same quantity; the
vectorized forms here use SciPy ``log_ndtr`` for array code paths.

Faddeeva / scaled complementary error function
----------------------------------------------
LFM and SIM covariances (``EQ_ODE1``, ``EQ_ODE2``) evaluate Upsilon kernels via
``wofz`` (Faddeeva ``w(z)``) and ``erfcx``. Those are the complex / scaled-erf
extensions of the same Gaussian-tail arithmetic.

**SciPy status:** ``scipy.special.wofz`` and ``erfcx`` are first-class SciPy
APIs. SciPy replaced the old ACM TOMS 680 Fortran ``WOFZ`` with Steven G.
Johnson's MIT-licensed Faddeeva C++ package in SciPy 0.12 (2012) for licensing
and accuracy. GPy has no Fortran Faddeeva of its own; the orphaned
``kern/src/sympy_helpers.cpp`` ``erfcx`` belongs to the removed symbolic-AD
stack (CHANGELOG / CIP-000E) and is not built.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx, log_ndtr, wofz

__all__ = [
    "wofz",
    "erfcx",
    "inv_mills_ratio",
    "d_inv_mills_ratio",
    "d2_inv_mills_ratio",
]


def inv_mills_ratio(z):
    """
    Vectorized inverse Mills ratio ``φ(z)/Φ(z)``.

    Evaluated as ``exp(log φ(z) - log Φ(z))`` with ``log_ndtr`` so the left
    tail stays finite (same pattern as Bernoulli–Probit).
    """
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z))


def d_inv_mills_ratio(z, lam=None):
    """Derivative of ``λ(z)=φ(z)/Φ(z)``: ``λ'(z) = -λ(z)(z + λ(z))``."""
    if lam is None:
        lam = inv_mills_ratio(z)
    return -lam * (z + lam)


def d2_inv_mills_ratio(z, lam=None):
    """
    Second derivative of ``λ(z)``.

    With ``λ' = -λ(z+λ)``,
    ``λ'' = -λ' (z+2λ) - λ = λ [(z+λ)(z+2λ) - 1]``.
    """
    if lam is None:
        lam = inv_mills_ratio(z)
    return lam * ((z + lam) * (z + 2.0 * lam) - 1.0)
