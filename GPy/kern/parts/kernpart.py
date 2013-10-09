# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


class Kernpart(object):
    def __init__(self,input_dim):
        """
        The base class for a kernpart: a positive definite function which forms part of a covariance function (kernel).

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        # the input dimensionality for the covariance
        self.input_dim = input_dim
        # the number of optimisable parameters
        self.num_params = 1
        # the name of the covariance function.
        self.name = 'unnamed'

    def _get_params(self):
        raise NotImplementedError
    def _set_params(self,x):
        raise NotImplementedError
    def _get_param_names(self):
        raise NotImplementedError
    def K(self,X,X2,target):
        raise NotImplementedError
    def Kdiag(self,X,target):
        raise NotImplementedError
    def dK_dtheta(self,dL_dK,X,X2,target):
        raise NotImplementedError
    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        # In the base case compute this by calling dK_dtheta. Need to
        # override for stationary covariances (for example) to save
        # time.
        for i in range(X.shape[0]):
            self.dK_dtheta(dL_dKdiag[i], X[i, :][None, :], X2=None, target=target)
    def psi0(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dtheta(self,dL_dpsi0,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dmuS(self,dL_dpsi0,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi1(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dtheta(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dZ(self,dL_dpsi1,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dmuS(self,dL_dpsi1,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi2(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dZ(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dtheta(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dmuS(self,dL_dpsi2,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def dK_dX(self, dL_dK, X, X2, target):
        raise NotImplementedError
    def dKdiag_dX(self, dL_dK, X, target):
        raise NotImplementedError



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
        self._X, self._X2, self._params = np.empty(shape=(3, 1))

    def _set_params(self, x):
        self.lengthscale = x
        self.lengthscale2 = np.square(self.lengthscale)
        # reset cached results
        self._X, self._X2, self._params = np.empty(shape=(3, 1))
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
        self._X, self._X2, self._params = np.empty(shape=(3, 1))


