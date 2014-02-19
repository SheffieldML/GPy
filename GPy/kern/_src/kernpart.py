# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#from ...core.parameterized.Parameterized import set_as_parameter
from ...core.parameterization import Parameterized

class Kernpart_stationary(Kernpart):
    def __init__(self, input_dim, lengthscale=None, ARD=False):
        self.input_dim = input_dim
        self.ARD = ARD
        if not ARD:
            self.num_params = 2
            if lengthscale is not None:
                self.lengthscale = np.asarray(lengthscale)
                assert self.lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                self.lengthscale = np.ones(1)
        else:
            self.num_params = self.input_dim + 1
            if lengthscale is not None:
                self.lengthscale = np.asarray(lengthscale)
                assert self.lengthscale.size == self.input_dim, "bad number of lengthscales"
            else:
                self.lengthscale = np.ones(self.input_dim)

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._parameters_ = np.empty(shape=(3, 1))

    def _set_params(self, x):
        self.lengthscale = x
        self.lengthscale2 = np.square(self.lengthscale)
        # reset cached results
        self._X, self._X2, self._parameters_ = np.empty(shape=(3, 1))
        self._Z, self._mu, self._S = np.empty(shape=(3, 1)) # cached versions of Z,mu,S


    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        # For stationary covariances, derivative of diagonal elements
        # wrt lengthscale is 0.
        target[0] += np.sum(dL_dKdiag)

    def dKdiag_dX(self, dL_dK, X, target):
        pass # true for all stationary kernels


class Kernpart_inner(Kernpart):
    def __init__(self,input_dim):
        """
        The base class for a kernpart_inner: a positive definite function which forms part of a kernel that is based on the inner product between inputs.

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        Kernpart.__init__(self, input_dim)

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._parameters_ = np.empty(shape=(3, 1))
