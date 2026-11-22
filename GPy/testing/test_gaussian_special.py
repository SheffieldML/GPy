# Copyright (c) 2026 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Tests for shared Gaussian special functions (Mills + SciPy Faddeeva)."""

import numpy as np
import pytest
from scipy.special import erfcx as scipy_erfcx
from scipy.special import log_ndtr
from scipy.special import wofz as scipy_wofz

from GPy.util.gaussian_special import (
    d2_inv_mills_ratio,
    d_inv_mills_ratio,
    erfcx,
    inv_mills_ratio,
    wofz,
)


def test_inv_mills_matches_log_ndtr_form():
    z = np.linspace(-4.0, 3.0, 50)
    expected = np.exp(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi) - log_ndtr(z))
    np.testing.assert_allclose(inv_mills_ratio(z), expected, rtol=1e-12)


def test_d_inv_mills_finite_difference():
    z = np.array([-2.0, -0.5, 0.0, 1.0, 2.5])
    eps = 1e-6
    fd = (inv_mills_ratio(z + eps) - inv_mills_ratio(z - eps)) / (2 * eps)
    np.testing.assert_allclose(d_inv_mills_ratio(z), fd, rtol=1e-5, atol=1e-6)


def test_d2_inv_mills_finite_difference():
    z = np.array([-1.5, 0.0, 1.0])
    eps = 1e-6
    fd = (d_inv_mills_ratio(z + eps) - d_inv_mills_ratio(z - eps)) / (2 * eps)
    np.testing.assert_allclose(d2_inv_mills_ratio(z), fd, rtol=1e-5, atol=1e-6)


def test_wofz_erfcx_are_scipy():
    assert wofz is scipy_wofz
    assert erfcx is scipy_erfcx
    z = np.array([0.0 + 0.0j, 1.0 + 0.5j])
    np.testing.assert_allclose(wofz(z), scipy_wofz(z))
    np.testing.assert_allclose(erfcx(np.array([0.5, 2.0])), scipy_erfcx(np.array([0.5, 2.0])))
