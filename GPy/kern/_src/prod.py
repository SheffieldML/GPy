# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from kern import CombinationKernel
from ...util.caching import Cache_this
import itertools

class Prod(CombinationKernel):
    """
    Computes the product of 2 kernels

    :param k1, k2: the kernels to multiply
    :type k1, k2: Kern
    :param tensor: The kernels are either multiply as functions defined on the same input space (default) or on the product of the input spaces
    :type tensor: Boolean
    :rtype: kernel object

    """
    def __init__(self, kernels, name='prod'):
        super(Prod, self).__init__(kernels, name)

    @Cache_this(limit=2, force_kwargs=['which_parts'])
    def K(self, X, X2=None, which_parts=None):
        assert X.shape[1] == self.input_dim
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.multiply, (p.K(X, X2) for p in which_parts))

    @Cache_this(limit=2, force_kwargs=['which_parts'])
    def Kdiag(self, X, which_parts=None):
        assert X.shape[1] == self.input_dim
        if which_parts is None:
            which_parts = self.parts
        return reduce(np.multiply, (p.Kdiag(X) for p in which_parts))

    def update_gradients_full(self, dL_dK, X):
        for k1,k2 in itertools.combinations(self.parts, 2):
            k1._sliced_X = k1._sliced_X2 = k2._sliced_X = k2._sliced_X2 = True
            k1.update_gradients_full(dL_dK*k2.K(X, X))
            self.k2.update_gradients_full(dL_dK*self.k1.K(X[:,self.slice1]), X[:,self.slice2])

    def gradients_X(self, dL_dK, X, X2=None):
        target = np.zeros(X.shape)
        if X2 is None:
            target[:,self.slice1] += self.k1.gradients_X(dL_dK*self.k2.K(X[:,self.slice2]), X[:,self.slice1], None)
            target[:,self.slice2] += self.k2.gradients_X(dL_dK*self.k1.K(X[:,self.slice1]), X[:,self.slice2], None)
        else:
            target[:,self.slice1] += self.k1.gradients_X(dL_dK*self.k2.K(X[:,self.slice2], X2[:,self.slice2]), X[:,self.slice1], X2[:,self.slice1])
            target[:,self.slice2] += self.k2.gradients_X(dL_dK*self.k1.K(X[:,self.slice1], X2[:,self.slice1]), X[:,self.slice2], X2[:,self.slice2])
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape)
        target[:,self.slice1] = self.k1.gradients_X(dL_dKdiag*self.k2.Kdiag(X[:,self.slice2]), X[:,self.slice1])
        target[:,self.slice2] += self.k2.gradients_X(dL_dKdiag*self.k1.Kdiag(X[:,self.slice1]), X[:,self.slice2])
        return target


