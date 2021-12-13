# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
The module for some general debug tools
"""

import numpy as np


def checkFinite(arr, name=None):
    if name is None:
        name = "Array with ID[" + str(id(arr)) + "]"

    if np.any(np.logical_not(np.isfinite(arr))):
        idx = np.where(np.logical_not(np.isfinite(arr)))[0]
        print(
            name
            + " at indices "
            + str(idx)
            + " have not finite values: "
            + str(arr[idx])
            + "!"
        )
        return False
    return True


def checkFullRank(m, tol=1e-10, name=None, force_check=False):
    if name is None:
        name = "Matrix with ID[" + str(id(m)) + "]"
    assert (
        len(m.shape) == 2 and m.shape[0] == m.shape[1]
    ), "The input of checkFullRank has to be a square matrix!"

    if not force_check and m.shape[0] >= 10000:  # pragma: no cover
        print("The size of " + name + "is too big to check (>=10000)!")
        return True

    s = np.real(np.linalg.eigvals(m))

    if s.min() / s.max() < tol:
        print(name + " is close to singlar!")
        print("The eigen values of " + name + " is " + str(s))
        return False
    return True
