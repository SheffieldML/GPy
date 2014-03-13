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
        assert len(kernels) == 2, 'only implemented for two kernels as of yet'
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

    def update_gradients_full(self, dL_dK, X, X2=None):
        for k1,k2 in itertools.combinations(self.parts, 2):
            k1.update_gradients_full(dL_dK*k2.K(X, X2), X, X2)
            k2.update_gradients_full(dL_dK*k1.K(X, X2), X, X2)

    def update_gradients_diag(self, dL_dKdiag, X):
        for k1,k2 in itertools.combinations(self.parts, 2):
            k1.update_gradients_diag(dL_dKdiag*k2.Kdiag(X), X)
            k2.update_gradients_diag(dL_dKdiag*k1.Kdiag(X), X)

    def gradients_X(self, dL_dK, X, X2=None):
        target = np.zeros(X.shape)
        for k1,k2 in itertools.combinations(self.parts, 2):
            target[:,k1.active_dims] += k1.gradients_X(dL_dK*k2.K(X, X2), X, X2)
            target[:,k2.active_dims] += k2.gradients_X(dL_dK*k1.K(X, X2), X, X2)
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape)
        for k1,k2 in itertools.combinations(self.parts, 2):
            target[:,k1.active_dims] += k1.gradients_X(dL_dKdiag*k2.Kdiag(X), X)
            target[:,k2.active_dims] += k2.gradients_X(dL_dKdiag*k1.Kdiag(X), X)
        return target


