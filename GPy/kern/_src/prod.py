# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from kern import CombinationKernel
from ...util.caching import Cache_this
import itertools


def numpy_invalid_op_as_exception(func):
    """
    A decorator that allows catching numpy invalid operations
    as exceptions (the default behaviour is raising warnings).
    """
    def func_wrapper(*args, **kwargs):
        np.seterr(invalid='raise')
        result = func(*args, **kwargs)
        np.seterr(invalid='warn')
        return result
    return func_wrapper


class Prod(CombinationKernel):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kern
    :param tensor: The kernels are either multiply as functions defined on the same input space (default) or on the product of the input spaces
    :type tensor: Boolean
    :rtype: kernel object

    """
    def __init__(self, kernels, name='mul'):
        for i, kern in enumerate(kernels[:]):
            if isinstance(kern, Prod):
                del kernels[i]
                for part in kern.parts[::-1]:
                    kern.unlink_parameter(part)
                    kernels.insert(i, part)
        super(Prod, self).__init__(kernels, name)

    @Cache_this(limit=2, force_kwargs=['which_parts'])
    def K(self, X, X2=None, which_parts=None):
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.multiply, (p.K(X, X2) for p in which_parts))

    @Cache_this(limit=2, force_kwargs=['which_parts'])
    def Kdiag(self, X, which_parts=None):
        if which_parts is None:
            which_parts = self.parts
        return reduce(np.multiply, (p.Kdiag(X) for p in which_parts))

    @numpy_invalid_op_as_exception
    def update_gradients_full(self, dL_dK, X, X2=None):
        k = self.K(X,X2)*dL_dK
        try:
            for p in self.parts:
                p.update_gradients_full(k/p.K(X,X2),X,X2)
        except FloatingPointError:
            for combination in itertools.combinations(self.parts, len(self.parts) - 1):
                prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
                to_update = list(set(self.parts) - set(combination))[0]
                to_update.update_gradients_full(dL_dK * prod, X, X2)

    def update_gradients_diag(self, dL_dKdiag, X):
        k = self.Kdiag(X)*dL_dKdiag
        for p in self.parts:
            p.update_gradients_diag(k/p.Kdiag(X),X)

    @numpy_invalid_op_as_exception            
    def gradients_X(self, dL_dK, X, X2=None):
        target = np.zeros(X.shape)
        k = self.K(X,X2)*dL_dK
        try:
            for p in self.parts:
                target += p.gradients_X(k/p.K(X,X2),X,X2)
        except FloatingPointError:
            for combination in itertools.combinations(self.parts, len(self.parts) - 1):
                prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
                to_update = list(set(self.parts) - set(combination))[0]
                target += to_update.gradients_X(dL_dK * prod, X, X2)
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape)
        k = self.Kdiag(X)*dL_dKdiag
        for p in self.parts:
            target += p.gradients_X_diag(k/p.Kdiag(X),X)
        return target


