# Copyright (c) 2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""User-facing validation should raise ValueError, not AssertionError (#922)."""

import numpy as np
import pytest

import GPy
from GPy.util.block_matrices import get_blocks


def test_gp_rejects_1d_X_with_value_error():
    X = np.linspace(0, 1, 5)
    Y = np.sin(X)[:, None]
    k = GPy.kern.RBF(1)
    with pytest.raises(ValueError, match="X must be 2-dimensional"):
        GPy.models.GPRegression(X, Y, k)


def test_gp_rejects_non_kernel_with_value_error():
    X = np.linspace(0, 1, 5)[:, None]
    Y = np.sin(X)
    with pytest.raises(ValueError, match="kernel must be"):
        GPy.models.GPRegression(X, Y, kernel="rbf")


def test_get_blocks_rejects_nonsquare_with_value_error():
    A = np.zeros((3, 2))
    with pytest.raises(ValueError, match="non-square"):
        get_blocks(A, [1, 2])
